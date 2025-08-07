// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#ifndef KINESIN_H
#define KINESIN_H

#include "digit.h"
class KinesinProp;


/// A model of the kinesin motor with discrete stepping
/**
 THIS CLASS IS A STUB and should not be used!
 
 Kinesin is derived from Digit, and it makes discrete jumps along the fiber.
 
 Stepping is stochastic.
 The size of the step size is set by @ref DigitPar digit:step_size.
 As defined by digit:use_lattice, the lattice may prevent binding & motion.
 As defined in Hand, detachment increases exponentially with force.
 
 See Examples and the @ref KinesinPar.
 @ingroup HandGroup
 
 @todo implement Kinesin
*/
class Kinesin : public Digit
{
private:
    
    /// disabled default constructor
    Kinesin();
    
    /// Gillespie countdown timer for stepping
    real   nextStep;

public:
    
    /// Property
    KinesinProp const* prop;
    
    /// constructor
    Kinesin(KinesinProp const* p, HandMonitor* h);
    
    /// destructor
    ~Kinesin() {}
    
    /// simulate when `this` is attached but not under load
    void   stepUnloaded();
    
    /// simulate when `this` is attached and under load
    void   stepLoaded(Vector const & force);
    
};

#endif

