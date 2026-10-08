# Two-story steel MRF (Kolay & Ricles). Units: kN, m, s.

proc createTwoStoryModel {} {
    set kN 1.0
    set m 1.0
    set s 1.0
    set kip 4.44822
    set inch 0.0254
    set foot [expr {12.0 * $inch}]
    set lbf [expr {$kip / 1000.0}]
    global g
    set g 9.80665
    set kg [expr {$kN / 1000.0 * $s * $s / $m}]
    set MPa 1000.0
    set GPa 1000000.0

    wipe
    model basic -ndm 2 -ndf 3

    set L 6.0
    set H 3.0
    set Lb [expr {$L / 12.0}]
    set Hc [expr {$H / 6.0}]

    node 1 0 [expr {2 * $H}]
    node 2 $Lb [expr {2 * $H}]
    node 3 [expr {2 * $Lb}] [expr {2 * $H}]
    node 4 [expr {0.5 * $L}] [expr {2 * $H}]
    node 5 [expr {$L - 2 * $Lb}] [expr {2 * $H}]
    node 6 [expr {$L - $Lb}] [expr {2 * $H}]
    node 7 $L [expr {2 * $H}]
    node 8 [expr {$L + 0.2 * $L}] [expr {2 * $H}]

    node 9 0 $H
    node 10 $Lb $H
    node 11 [expr {2 * $Lb}] $H
    node 12 [expr {0.5 * $L}] $H
    node 13 [expr {$L - 2 * $Lb}] $H
    node 14 [expr {$L - $Lb}] $H
    node 15 $L $H
    node 16 [expr {$L + 0.2 * $L}] $H

    node 17 0 0
    node 18 $L 0
    node 19 [expr {$L + 0.2 * $L}] 0

    set z0 [expr {($H - 2 * $Hc) / 2.0}]
    set z1 [expr {$H - 2 * $Hc}]
    set z2 [expr {$H - $Hc}]
    set z3 [expr {$H + $Hc}]
    set z4 [expr {$H + 2 * $Hc}]
    set z5 [expr {2 * $H - 2 * $Hc}]
    set z6 [expr {2 * $H - $Hc}]

    node 20 0 $z0
    node 21 0 $z1
    node 22 0 $z2
    node 23 0 $z3
    node 24 0 $z4
    node 25 0 $z5
    node 26 0 $z6
    node 27 $L $z0
    node 28 $L $z1
    node 29 $L $z2
    node 30 $L $z3
    node 31 $L $z4
    node 32 $L $z5
    node 33 $L $z6

    fix 17 1 1 0
    fix 18 1 1 0
    fix 19 1 1 0
    rigidDiaphragm 1 8 4
    rigidDiaphragm 1 16 12

    set mFloor [expr {50.97e3 * $kg}]
    mass 8 $mFloor $kg [expr {1.0 * $kg * $m}]
    mass 16 $mFloor $kg [expr {1.0 * $kg * $m}]

    uniaxialMaterial Steel01 1 [expr {345.0 * $MPa}] [expr {200.0 * $GPa}] 0.01
    section WFSection2d 1 1 [expr {23.6 * $inch}] [expr {0.395 * $inch}] \
        [expr {7.01 * $inch}] [expr {0.505 * $inch}] 10 3
    section WFSection2d 2 1 [expr {14.5 * $inch}] [expr {0.59 * $inch}] \
        [expr {14.7 * $inch}] [expr {0.94 * $inch}] 10 3

    geomTransf Linear 1
    beamIntegration Legendre 1 1 2
    set rhoB [expr {(55.0 * $lbf / $foot) / $g}]
    element dispBeamColumn 1 1 2 1 1 -cMass -mass $rhoB
    element dispBeamColumn 2 2 3 1 1 -cMass -mass $rhoB
    element dispBeamColumn 3 3 4 1 1 -cMass -mass $rhoB
    element dispBeamColumn 4 4 5 1 1 -cMass -mass $rhoB
    element dispBeamColumn 5 5 6 1 1 -cMass -mass $rhoB
    element dispBeamColumn 6 6 7 1 1 -cMass -mass $rhoB
    element dispBeamColumn 7 9 10 1 1 -cMass -mass $rhoB
    element dispBeamColumn 8 10 11 1 1 -cMass -mass $rhoB
    element dispBeamColumn 9 11 12 1 1 -cMass -mass $rhoB
    element dispBeamColumn 10 12 13 1 1 -cMass -mass $rhoB
    element dispBeamColumn 11 13 14 1 1 -cMass -mass $rhoB
    element dispBeamColumn 12 14 15 1 1 -cMass -mass $rhoB

    geomTransf Linear 2
    beamIntegration Legendre 2 2 2
    set rhoC [expr {(120.0 * $lbf / $foot) / $g}]
    element dispBeamColumn 15 17 20 2 2 -cMass -mass $rhoC
    element dispBeamColumn 16 20 21 2 2 -cMass -mass $rhoC
    element dispBeamColumn 17 21 22 2 2 -cMass -mass $rhoC
    element dispBeamColumn 18 22 9 2 2 -cMass -mass $rhoC
    element dispBeamColumn 19 9 23 2 2 -cMass -mass $rhoC
    element dispBeamColumn 20 23 24 2 2 -cMass -mass $rhoC
    element dispBeamColumn 21 24 25 2 2 -cMass -mass $rhoC
    element dispBeamColumn 22 25 26 2 2 -cMass -mass $rhoC
    element dispBeamColumn 23 26 1 2 2 -cMass -mass $rhoC
    element dispBeamColumn 24 18 27 2 2 -cMass -mass $rhoC
    element dispBeamColumn 25 27 28 2 2 -cMass -mass $rhoC
    element dispBeamColumn 26 28 29 2 2 -cMass -mass $rhoC
    element dispBeamColumn 27 29 15 2 2 -cMass -mass $rhoC
    element dispBeamColumn 28 15 30 2 2 -cMass -mass $rhoC
    element dispBeamColumn 29 30 31 2 2 -cMass -mass $rhoC
    element dispBeamColumn 30 31 32 2 2 -cMass -mass $rhoC
    element dispBeamColumn 31 32 33 2 2 -cMass -mass $rhoC
    element dispBeamColumn 32 33 7 2 2 -cMass -mass $rhoC

    geomTransf PDelta 3
    set A [expr {9.76e-2 * $m * $m}]
    set I [expr {7.125e-4 * $m * $m * $m * $m}]
    set rhoG [expr {1e-3 * $kg / $m}]
    element elasticBeamColumn 33 19 16 $A [expr {200.0 * $GPa}] $I 3 -mass $rhoG -cMass
    element elasticBeamColumn 34 16 8 $A [expr {200.0 * $GPa}] $I 3 -mass $rhoG -cMass
}

proc applyGravity {} {
    global g
    timeSeries Linear 1
    pattern Plain 1 1 {
        load 8 0 -500.0 0
        load 16 0 -500.0 0
    }
    wipeAnalysis
    constraints Transformation
    numberer RCM
    system BandGeneral
    test NormDispIncr 1e-6 10
    algorithm Newton
    integrator LoadControl 0.1
    analysis Static
    if {[analyze 10] < 0} {
        error "gravity static analysis failed to converge"
    }
    loadConst -time 0.0
}
