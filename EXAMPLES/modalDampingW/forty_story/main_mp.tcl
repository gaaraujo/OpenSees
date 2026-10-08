# Partitioned OpenSeesMP driver for the 40-story Woodbury example.
# Contiguous story blocks via getPID/getNP; shared interface nodes.
#
# Run (from this directory, with oneAPI env loaded):
#   mpirun -np 2 ../../../../build/Release/OpenSeesMP main_mp.tcl
#
# Compares modalDampingQ / modalDamping -legacy / modalDamping -woodbury
# under Mumps and DistributedProfileSPD (system ParallelProfileSPD).
# Tracks Newton iterations and peak interstory drift vs uy so elastic
# steps are confirmed to converge in a single iteration.

set pid  [getPID]
set numP [getNP]

if {$numP < 2} {
    puts "ERROR: main_mp.tcl needs at least 2 ranks (np=$numP)"
    exit 1
}

set Nstories 40
set m 1.0
set kbottom 900.0
set ktop 600.0
set Nmodes 6
set zeta 0.02
set uy 0.02
set b 0.01
set dt 0.02
set g 9.81
set tabasScale 1.0
# Shortened vs serial main.tcl for a faster MP smoke; set Texc {} to use full record.
set Texc 2.0
set Tfree 0.0

set scriptDir [file dirname [info script]]
source [file join $scriptDir .. common.tcl]
set tabasFile [file join $scriptDir tabasFN.txt]
if {![file exists $tabasFile]} {
    error "ERROR: missing $tabasFile"
}

if {$Texc eq {}} {
    set fh [open $tabasFile r]
    set Nexc 0
    while {[gets $fh line] >= 0} {
        if {[string trim $line] ne ""} { incr Nexc }
    }
    close $fh
} else {
    set Nexc [expr {int($Texc / $dt)}]
}
set Nfree [expr {int($Tfree / $dt)}]

set resultsDir [file join $scriptDir results_mp]
set logsDir [file join $scriptDir logs_mp]

if {$pid == 0} {
    foreach d [list $resultsDir $logsDir] {
        if {[file exists $d]} { file delete -force $d }
        file mkdir $d
    }
}
barrier

# Contiguous element blocks: rank r owns stories [e0+1 .. e1]
# with nodes e0..e1; interface node e0 is shared with rank r-1.
proc storyRange {pid numP Nstories} {
    set base [expr {$Nstories / $numP}]
    set rem  [expr {$Nstories % $numP}]
    set e0 0
    for {set p 0} {$p < $pid} {incr p} {
        set nHere [expr {$base + ($p < $rem ? 1 : 0)}]
        incr e0 $nHere
    }
    set nLocal [expr {$base + ($pid < $rem ? 1 : 0)}]
    set e1 [expr {$e0 + $nLocal}]
    return [list $e0 $e1]
}

proc buildPartitionedModel {pid numP Nstories m kbottom ktop uy b} {
    model basic -ndm 1 -ndf 1

    lassign [storyRange $pid $numP $Nstories] e0 e1
    for {set n $e0} {$n <= $e1} {incr n} {
        node $n 0
        if {$n == 0} {
            fix 0 1
        } elseif {$n > $e0} {
            mass $n $m
        }
    }

    for {set e [expr {$e0 + 1}]} {$e <= $e1} {incr e} {
        set k [expr {$kbottom + ($ktop - $kbottom) * double($e - 1) / double($Nstories - 1)}]
        set fy [expr {$k * $uy}]
        uniaxialMaterial Steel01 $e $fy $k $b
        element zeroLength $e [expr {$e - 1}] $e -mat $e -dir 1
    }

    set localNodes {}
    for {set n $e0} {$n <= $e1} {incr n} {
        lappend localNodes $n
    }
    return [list $e0 $e1 $localNodes]
}

# Peak |interstory| on this rank (story e uses nodes e-1 and e).
proc localPeakDrift {e0 e1} {
    set peak 0.0
    for {set e [expr {$e0 + 1}]} {$e <= $e1} {incr e} {
        set du [expr {abs([nodeDisp $e 1] - [nodeDisp [expr {$e - 1}] 1])}]
        if {$du > $peak} { set peak $du }
    }
    return $peak
}

set cases {
    {modalDampingQ}
    {modalDamping}
    {modalDampingW}
}

# ParallelProfileSPD -> DistributedProfileSPDLinSOE (ProfileSPD alone is serial)
set solvers {
    {Mumps Mumps}
    {ParallelProfileSPD DistProfileSPD}
}

lassign [storyRange $pid $numP $Nstories] myE0 myE1
if {$pid == 0} {
    puts "40-story MP partition np=$numP  Nmodes=$Nmodes  Nexc=$Nexc  uy=$uy"
}
puts "  rank $pid owns stories ([expr {$myE0+1}])..$myE1  nodes $myE0..$myE1"
barrier

foreach solverRow $solvers {
    lassign $solverRow systemCmd solverLabel

    foreach row $cases {
        lassign $row caseTag
        set tag ${caseTag}_${solverLabel}_np${numP}

        if {$pid == 0} {
            puts "=== $tag ==="
        }
        barrier

        logFile [file join $logsDir opensees_${tag}_rank${pid}.log] -noEcho

        wipe
        lassign [buildPartitionedModel $pid $numP $Nstories $m $kbottom $ktop $uy $b] e0 e1 localNodes

        constraints Plain
        numberer ParallelPlain
        system $systemCmd
        test NormUnbalance 1e-8 25 0
        algorithm Newton
        integrator Newmark 0.5 0.25
        analysis Transient -noWarnings

        set okEigen [eigen $Nmodes]
        if {$pid == 0} {
            puts "  eigen ok, omega^2(1)=[lindex $okEigen 0]"
        }
        barrier

        applyModalDamping $caseTag [lrepeat $Nmodes $zeta]

        timeSeries Path 1 -filePath $tabasFile -dt $dt -factor [expr {$g * $tabasScale}]
        pattern UniformExcitation 1 1 -accel 1

        set recordNodes {}
        foreach n $localNodes {
            if {$n != 0 && $n > $e0} {
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
        set maxIters 0
        set minIters 999
        set maxDriftLocal 0.0
        set fhConv [open [file join $resultsDir ${tag}_rank${pid}_convergence.dat] w]
        puts $fhConv "# time iters final_norm peak_local_drift"

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
            set iters [testIter]
            set norms [testNorms]
            if {$iters > $maxIters} { set maxIters $iters }
            if {$iters < $minIters} { set minIters $iters }
            set drift [localPeakDrift $e0 $e1]
            if {$drift > $maxDriftLocal} { set maxDriftLocal $drift }
            set t [expr {$step * $dt}]
            puts $fhConv "$t $iters [finalNorm $iters $norms] $drift"
        }
        close $fhConv
        set elapsed [expr {([clock milliseconds] - $t0) / 1000.0}]

        # Allreduce-ish peak drift via files (simple + portable for this smoke).
        set fhD [open [file join $resultsDir ${tag}_rank${pid}_peakdrift.txt] w]
        puts $fhD $maxDriftLocal
        close $fhD
        barrier

        set maxDriftGlobal $maxDriftLocal
        if {$pid == 0} {
            for {set p 0} {$p < $numP} {incr p} {
                set f [file join $resultsDir ${tag}_rank${p}_peakdrift.txt]
                if {[file exists $f]} {
                    set fh [open $f r]
                    gets $fh d
                    close $fh
                    if {$d > $maxDriftGlobal} { set maxDriftGlobal $d }
                }
            }
            set elastic [expr {$maxDriftGlobal < $uy}]
            if {$failed} {
                puts "  $tag: FAILED after $nDone/$Nsteps steps ([format %.2f $elapsed] s)"
            } else {
                puts "  $tag: successful ($nDone steps, [format %.2f $elapsed] s)"
                puts "    Newton iters min/max=$minIters/$maxIters  peakDrift=[format %.6g $maxDriftGlobal]  uy=$uy  elastic=$elastic"
                if {$elastic && $maxIters != 1} {
                    puts "    WARNING: elastic range but max Newton iters=$maxIters (expected 1)"
                } elseif {$elastic && $maxIters == 1} {
                    puts "    OK: single Newton iteration throughout elastic range"
                }
            }
        }
        barrier
    }
}

if {$pid == 0} {
    puts "Done. Logs: $logsDir  Results: $resultsDir"
}
