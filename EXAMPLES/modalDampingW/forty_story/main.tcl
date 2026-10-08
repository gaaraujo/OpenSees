# 40-story shear building — modalDampingW example
# Run: ../../../../build/Release/OpenSees main.tcl
#      python3 plotResults.py

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
set TfreeFrac 1.0

set scriptDir [file dirname [info script]]
source [file join $scriptDir .. common.tcl]
set tabasFile [file join $scriptDir tabasFN.txt]
if {![file exists $tabasFile]} {
    error "ERROR: missing $tabasFile"
}
set fh [open $tabasFile r]
set Nexc 0
while {[gets $fh line] >= 0} {
    if {[string trim $line] ne ""} {
        incr Nexc
    }
}
close $fh
set Nfree [expr {int($TfreeFrac * $Nexc)}]
set resultsDir [file join $scriptDir results]
set logsDir [file join $scriptDir logs]
set figuresDir [file join $scriptDir figures]

resetOutputDirs $scriptDir

set cases {
    {modalDampingQ BandGeneral}
    {modalDamping BandGeneral}
    {modalDampingW BandGeneral}
    {modalDamping FullGeneral}
}

set recordNodes {0}
for {set i 1} {$i <= $Nstories} {incr i} {
    lappend recordNodes $i
}

foreach row $cases {
    lassign $row caseTag system
    set tag ${caseTag}_${system}

    logFile [file join $logsDir opensees_${tag}.log] -noEcho

    wipe
    model basic -ndm 1 -ndf 1

    node 0 0; fix 0 1
    for {set i 1} {$i <= $Nstories} {incr i} {
        node $i 0
        mass $i $m
        set k [expr {$kbottom + ($ktop - $kbottom) * double($i - 1) / double($Nstories - 1)}]
        set fy [expr {$k * $uy}]
        uniaxialMaterial Steel01 $i $fy $k $b
        element zeroLength $i [expr {$i - 1}] $i -mat $i -dir 1
    }

    eigen $Nmodes
    applyModalDamping $caseTag [lrepeat $Nmodes $zeta]

    timeSeries Path 1 -filePath $tabasFile -dt $dt -factor [expr {$g * $tabasScale}]
    pattern UniformExcitation 1 1 -accel 1

    numberer Plain
    system $system
    test NormUnbalance 1e-8 10 1
    analysis Transient -noWarnings

    foreach resp {disp vel accel} {
        recorder Node -file [file join $resultsDir ${tag}_${resp}.out] \
            -time -node {*}$recordNodes -dof 1 $resp
    }

    analyzeSteps $tag $dt $Nexc $Nfree $resultsDir
}

oldputs "Done. Run: python3 plotResults.py to see results."
