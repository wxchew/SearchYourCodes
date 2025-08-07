// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef WALKER_H
#define WALKER_H

#include "digit.h"
class WalkerProp;


/// A Hand that move with discrete steps of fixed size along a Fiber
/**
 The Walker moves like a Motor, but it makes discrete jumps along the fiber.
 
 The size of the step size is set by @ref DigitPar digit:step_size.
 The number of steps in one `time_step` is a stochastic integer  
 with a Poisson distribution.
 
 As defined in Hand, detachment increases exponentially with force.

 If @ref DigitPar digit:use_lattice = 1, the Digit will use the Lattice
 associated to the fiber to limit the occupancy of a lattice site to 1:
 - the digit will bind only at an empty lattice site,
 - it will not step into an occupied site,
 - upon binding, the digit will occupy the corresponding site entirely
 .
 
 See Examples and the @ref WalkerPar.
 @ingroup HandGroup 

 Note: Conventional kinesin can make backward steps under heavy load:\n
 Carter, N. & Cross, R. Mechanics of the kinesin step. Nature 435, 308–312 (2005).
 According to this, a real kinesin stalls because the probability of making a forward step, 
 which is decreased by load, is equal to the probability of making a backward step. 

 With antagonistic force however, the Walker will stall in a immobile configuration,
 because it can only make forward steps, wereas a stalled kinesin keeps moving
 back-and-forth over a few lattice sites.
 Thus one should be careful when using Walker to model Kinesin.

 @todo handle cases where digit::step_size = INT * fiber:lattice_size
*/
class Walker : public Digit
{
private:
    
    /// disabled default constructor
    Walker();
    
    /// indicates directionality, this should be a property!
//    bool   plusEndDirected;
    
    /// Gillespie countdown timer for stepping
    real   nextStep;
    
    /// attempts a step towards PLUS_END if ( walker:max_speed > 0 )
    int  stepForward();
    
    /// attempts a step towards MINUS_END if ( walker:max_speed > 0 )
    int  stepBackward();
    
    /// Checks whether the next place is free
    int  checkNext();
    
    /// Value of the step, this should be a property!
//    int step_val;
public:
    
    /// Property
    WalkerProp const* prop;
    
    /// change directionality
    void setDirectionality(bool plus);

    /// constructor
    Walker(WalkerProp const* p, HandMonitor* h);
    
    /// destructor
    ~Walker() {}

    /// attach and update variables
    void         attach(FiberBinder const& site);
    
    /// simulate when `this` is attached but not under load
    void         stepUnloaded();
    
    /// simulate when `this` is attached and under load
    void         stepLoaded(Vector const & force);
    
};

#endif

