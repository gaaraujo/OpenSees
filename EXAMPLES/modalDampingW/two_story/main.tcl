# Two-story steel MRF — modalDampingW example
# Run: ../../../../build/Release/OpenSees main.tcl
#      python3 plotResults.py

set Nmodes 2
set zeta 0.02
set dt 0.01
set gmScale 3.0
set TfreeFrac 1.0
set gmPatternTag 2

set scriptDir [file dirname [info script]]
source [file join $scriptDir .. common.tcl]
source [file join $scriptDir model.tcl]

set gmFile [file join $scriptDir RSN960_NORTHR_LOS270.txt]
if {![file exists $gmFile]} {
    error "ERROR: missing $gmFile"
}
set fh [open $gmFile r]
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
set recordNodes {17 16 8}

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

    createTwoStoryModel
    applyGravity
    eigen $Nmodes
    applyModalDamping $caseTag [list $zeta $zeta]

    timeSeries Path $gmPatternTag -filePath $gmFile -dt $dt \
        -factor [expr {$g * $gmScale}]
    pattern UniformExcitation $gmPatternTag 1 -accel $gmPatternTag

    wipeAnalysis
    constraints Transformation
    numberer RCM
    system $system
    test NormUnbalance 1e-8 50 2
    algorithm Newton
    integrator Newmark 0.5 0.25
    analysis Transient -noWarnings

    foreach resp {disp vel accel} {
        recorder Node -file [file join $resultsDir ${tag}_${resp}.out] \
            -time -node {*}$recordNodes -dof 1 $resp
    }

    analyzeSteps $tag $dt $Nexc $Nfree $resultsDir $gmPatternTag
}

oldputs "Done. Run: python3 plotResults.py to see results."
