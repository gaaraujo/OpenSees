# Shared Tcl helpers for modal damping example drivers (not model geometry).

proc applyModalDamping {caseTag zetas} {
    switch $caseTag {
        modalDampingQ {
            eval modalDampingQ $zetas
        }
        modalDamping {
            eval modalDamping -legacy $zetas
        }
        modalDampingW {
            eval modalDamping -woodbury $zetas
        }
        default {
            error "unknown modal damping case: $caseTag"
        }
    }
}

proc resetOutputDirs {root} {
    foreach sub {results logs figures} {
        set d [file join $root $sub]
        if {[file exists $d]} {
            file delete -force $d
        }
        file mkdir $d
    }
}

proc finalNorm {iters norms} {
    set n [llength $norms]
    if {$iters > 0 && $n >= $iters} {
        return [lindex $norms [expr {$iters - 1}]]
    }
    if {$n > 0} {
        return [lindex $norms end]
    }
    return 0.0
}

proc analyzeSteps {tag dt Nexc Nfree resultsDir {gmPat 1}} {
    set Nsteps [expr {$Nexc + $Nfree}]
    set t0 [clock milliseconds]
    set nDone 0
    set fh [open [file join $resultsDir ${tag}_convergence.dat] w]
    puts $fh "# time iters final_norm"
    for {set step 1} {$step <= $Nsteps} {incr step} {
        if {$step == [expr {$Nexc + 1}]} {
            remove loadPattern $gmPat
        }
        set ok [analyze 1 $dt]
        if {$ok < 0} {
            set elapsed [expr {([clock milliseconds] - $t0) / 1000.0}]
            oldputs "  $tag: analyze failed at step $step ([format %.2f $elapsed] s)"
            break
        }
        incr nDone
        set iters [testIter]
        set norms [testNorms]
        set t [expr {$step * $dt}]
        puts $fh "$t $iters [finalNorm $iters $norms]"
    }
    close $fh
    if {$nDone == $Nsteps} {
        set elapsed [expr {([clock milliseconds] - $t0) / 1000.0}]
        oldputs "  $tag: successful ([format %.2f $elapsed] s)"
    }
}
