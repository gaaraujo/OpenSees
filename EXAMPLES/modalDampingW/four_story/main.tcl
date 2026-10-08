# Four-story shear frame (modalDampingW example) — https://openseesdigital.com/2019/09/12/be-careful-with-modal-damping/
# Run: ../../../../build/Release/OpenSees main.tcl
#      python3 plotResults.py

set k 610.0
set m 1.0352
set zeta 0.02
set uy 0.02
set b 0.01
set fy [expr {$k * $uy}]
set dt 0.02
set g 9.81
set tabasScale 1.0
set Texc 10.0
set Tfree 10.0
set Nexc [expr {int($Texc / $dt)}]
set Nfree [expr {int($Tfree / $dt)}]

set scriptDir [file dirname [info script]]
source [file join $scriptDir .. common.tcl]
set tabasFile [file join $scriptDir tabasFN.txt]
if {![file exists $tabasFile]} {
    error "ERROR: missing $tabasFile"
}
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

foreach row $cases {
    lassign $row caseTag system
    set tag ${caseTag}_${system}

    logFile [file join $logsDir opensees_${tag}.log] -noEcho

    wipe
    model basic -ndm 1 -ndf 1

    node 0 0; fix 0 1
    node 1 0; mass 1 $m
    node 2 0; mass 2 $m
    node 3 0; mass 3 $m
    node 4 0; mass 4 [expr {0.5 * $m}]
    set recordNodes {0 1 2 3 4}

    uniaxialMaterial Steel01 1 $fy $k $b

    element zeroLength 1 0 1 -mat 1 -dir 1
    element zeroLength 2 1 3 -mat 1 -dir 1
    element zeroLength 3 3 2 -mat 1 -dir 1
    element zeroLength 4 3 4 -mat 1 -dir 1

    eigen 1
    applyModalDamping $caseTag [list $zeta]

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
