# Partitioned OpenSeesMP driver for the four-story Woodbury example.
# Build the shear frame by story ownership via getPID/getNP (shared interface nodes).
#
# Run (from this directory, with oneAPI env loaded):
#   mpirun -np 2 ../../../../build/Release/OpenSeesMP main_mp.tcl
#
# Compares modalDampingQ / modalDamping -legacy / modalDamping -woodbury under Mumps.

set pid  [getPID]
set numP [getNP]

if {$numP < 2} {
    puts "ERROR: main_mp.tcl needs at least 2 ranks (np=$numP)"
    exit 1
}

set k 610.0
set m 1.0352
set zeta 0.02
set uy 0.02
set b 0.01
set fy [expr {$k * $uy}]
set dt 0.02
set g 9.81
set tabasScale 1.0
set Texc 2.0
set Tfree 0.0
set Nexc [expr {int($Texc / $dt)}]
set Nfree [expr {int($Tfree / $dt)}]

set scriptDir [file dirname [info script]]
source [file join $scriptDir .. common.tcl]
set tabasFile [file join $scriptDir tabasFN.txt]
if {![file exists $tabasFile]} {
    error "ERROR: missing $tabasFile"
}
set resultsDir [file join $scriptDir results_mp]
set logsDir [file join $scriptDir logs_mp]

if {$pid == 0} {
    foreach d [list $resultsDir $logsDir] {
        if {[file exists $d]} { file delete -force $d }
        file mkdir $d
    }
}
barrier

# Story elements 1..4 connect nodes (i-1)--i with Scott wiring:
#   ele 1: 0-1, ele 2: 1-3, ele 3: 3-2, ele 4: 3-4
# Partition by element ownership across ranks (round-robin). Shared nodes
# are created on every rank that owns an incident element; mass is owned
# by the lowest-rank owner so Arpack/Mumps do not double-count.
proc storyNodes {ele} {
    switch $ele {
        1 { return {0 1} }
        2 { return {1 3} }
        3 { return {3 2} }
        4 { return {3 4} }
        default { error "bad ele $ele" }
    }
}

proc ownsElement {pid numP ele} {
    return [expr {(($ele - 1) % $numP) == $pid}]
}

proc massOwner {nodeTag numP} {
    # Lowest rank that owns an element incident to this node.
    for {set e 1} {$e <= 4} {incr e} {
        foreach n [storyNodes $e] {
            if {$n == $nodeTag} {
                for {set p 0} {$p < $numP} {incr p} {
                    if {[ownsElement $p $numP $e]} {
                        return $p
                    }
                }
            }
        }
    }
    return 0
}

proc buildPartitionedModel {pid numP k m fy b} {
    model basic -ndm 1 -ndf 1
    uniaxialMaterial Steel01 1 $fy $k $b

    array set needNode {}
    for {set e 1} {$e <= 4} {incr e} {
        if {[ownsElement $pid $numP $e]} {
            foreach n [storyNodes $e] {
                set needNode($n) 1
            }
        }
    }

    foreach n [lsort -integer [array names needNode]] {
        node $n 0
        if {$n == 0} {
            fix 0 1
        } elseif {[massOwner $n $numP] == $pid} {
            if {$n == 4} {
                mass $n [expr {0.5 * $m}]
            } else {
                mass $n $m
            }
        }
    }

    for {set e 1} {$e <= 4} {incr e} {
        if {[ownsElement $pid $numP $e]} {
            lassign [storyNodes $e] i j
            element zeroLength $e $i $j -mat 1 -dir 1
        }
    }

    set localNodes [lsort -integer [array names needNode]]
    return $localNodes
}

set cases {
    {modalDampingQ}
    {modalDamping}
    {modalDampingW}
}

foreach row $cases {
    lassign $row caseTag
    set tag ${caseTag}_Mumps_np${numP}

    if {$pid == 0} {
        puts "=== $tag (pid=$pid/$numP) ==="
    }
    barrier

    logFile [file join $logsDir opensees_${tag}_rank${pid}.log] -noEcho

    wipe
    set localNodes [buildPartitionedModel $pid $numP $k $m $fy $b]

    constraints Plain
    numberer ParallelPlain
    system Mumps
    test NormUnbalance 1e-8 25 0
    algorithm Newton
    integrator Newmark 0.5 0.25
    analysis Transient -noWarnings

    # Eigen after parallel SOE/numberer so Arpack gets Mumps channels.
    set nmodes 1
    set okEigen [eigen $nmodes]
    if {$pid == 0} {
        puts "  eigen ok, omega^2=$okEigen"
    }
    barrier

    applyModalDamping $caseTag [list $zeta]

    timeSeries Path 1 -filePath $tabasFile -dt $dt -factor [expr {$g * $tabasScale}]
    pattern UniformExcitation 1 1 -accel 1

    # Record only locally owned free nodes to avoid missing-node errors.
    set recordNodes {}
    foreach n $localNodes {
        if {$n != 0 && [massOwner $n $numP] == $pid} {
            lappend recordNodes $n
        }
    }
    if {[llength $recordNodes] > 0} {
        foreach resp {disp vel accel} {
            recorder Node -file [file join $resultsDir ${tag}_rank${pid}_${resp}.out] \
                -time -node {*}$recordNodes -dof 1 $resp
        }
    }

    set Nsteps [expr {$Nexc + $Nfree}]
    set t0 [clock milliseconds]
    set nDone 0
    set failed 0
    for {set step 1} {$step <= $Nsteps} {incr step} {
        if {$step == [expr {$Nexc + 1}]} {
            remove loadPattern 1
        }
        set ok [analyze 1 $dt]
        if {$ok < 0} {
            set failed 1
            if {$pid == 0} {
                puts "  $tag: analyze failed at step $step"
            }
            break
        }
        incr nDone
    }
    set elapsed [expr {([clock milliseconds] - $t0) / 1000.0}]
    if {$pid == 0} {
        if {$failed} {
            puts "  $tag: FAILED after $nDone/$Nsteps steps ([format %.2f $elapsed] s)"
        } else {
            puts "  $tag: successful ($nDone steps, [format %.2f $elapsed] s)"
        }
    }
    barrier
}

if {$pid == 0} {
    puts "Done. Logs: $logsDir  Results: $resultsDir"
}
