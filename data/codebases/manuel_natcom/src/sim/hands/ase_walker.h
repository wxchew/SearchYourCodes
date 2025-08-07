// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef ASE_WALKER_H
#define ASE_WALKER_H

#include "digit.h"

//#define EXPONENTIAL_ASE1
// The k+/k- ratio of Ase1 should satisfy detailed balance/arrhenius law. A simple exponential
// solution is:
// alpha = f*a/kT
// k+ = exp(alpha/2) and k- = exp(-alpha/2)
// Alternatively one can use the solution from Wang et al.
// WANG, H., PESKIN, C. S., & ELSTON, T. C. (2003). A Robust Numerical Algorithm for Studying Biomolecular Transport Processes. Journal of Theoretical Biology, 221(4), 491–511. https://doi.org/10.1006/jtbi.2003.3200


class AseWalkerProp;

class AseWalker : public Digit
{
private:
    
    /// disabled default constructor
    AseWalker();
    
    /// Gillespie countdown timer for stepping
    real   nextStep;
#ifdef MULTI_LATTICE
    /// Gillespie counter to move to neighbour lattice in multi_lattice
    real   nextRedist;
#endif
public:
    
    /// Property
    AseWalkerProp const* prop;

    /// constructor
    AseWalker(AseWalkerProp const* p, HandMonitor* h);
    
    /// destructor
    ~AseWalker() {}

    /// attach and update variables
    void         attach(FiberBinder const& site);
    
    /// simulate when `this` is attached but not under load
    void         stepUnloaded();
    
    /// simulate when `this` is attached and under load
    void         stepLoaded(Vector const & force);
    
    /// this is called when the attachment point is beyond the PLUS_END
    void   handleDisassemblyM();
    
    /// this is called when the attachment point is below the MINUS_END
    void   handleDisassemblyP();
    
    /// Calculate plus and minus ward propensities (k+ = k0 * p_plus, same for minus)
    void calc_propensities(Vector const& force, real& p_plus, real& p_min);
    
    /// Rate of movement to any direction (sum of all rates)
    real   propensUnloaded();

    /// Rate of movement to any direction (sum of all rates)
    real   propensLoaded(Vector const&);
    
    int   gillestepUnloaded();
    
    int   gillestepLoaded(Vector const & force);
    
    /// When there is a shrinking end, and hold_shrinking_end is true, then sweep the following
    /// ase1 hands to the right. This should not happen with a motor that uses the lattice.
    ///void  sweepP();
    
    // Maybe at some point implement this
    //void  sweepM();
    
    bool needs_sweep();
    
    void sweep();
    
    bool first_sweep();
    
    bool testDangling();
    
#ifdef DIGITS_GILLESPIE
    /// Set that firing time
//    real set_fire();
//    
//    real set_fire_force(real f);
#endif

};

#endif

