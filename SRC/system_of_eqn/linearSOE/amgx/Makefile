include ../../../../Makefile.def

OBJS       = 
ifdef AMGX
OBJS       = AmgXGenLinSOE.o AmgXGenLinSolver.o 
endif

all:         $(OBJS)

# Miscellaneous
tidy:	
	@$(RM) $(RMFLAGS) Makefile.bak *~ #*# core

clean: tidy
	@$(RM) $(RMFLAGS) $(OBJS) *.o

spotless: clean
	@$(RM) $(RMFLAGS)

wipe: spotless

# DO NOT DELETE THIS LINE -- make depend depends on it.
