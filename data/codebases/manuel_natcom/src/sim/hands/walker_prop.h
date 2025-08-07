// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef WALKER_PROP_H
#define WALKER_PROP_H

#include "digit_prop.h"


/// Additional Property for Walker
/**
 @ingroup Properties
*/
class WalkerProp : public DigitProp
{
    friend class Walker;
    
public:
    
    /**
     @defgroup WalkerPar Parameters of Walker
     @ingroup Parameters
     Inherits @ref DigitPar.
     @{
     */

    /// force at which stepping rate becomes zero
    real    stall_force;
    
    /// speed if `force=0` ( `max_speed = rate * step_size` )
    /**
     A positive value specifies a plus-end directed motor.
     A negative value specifies a minus-end directed motor.
     */
    real    unloaded_speed;
    
    /// probability to unbind per step
    /**
     This probability in [0,1] is tested for every successful step,
     and thus creates an additional effective detachment rate equal to
     `unbinding_chance * ( max_speed / step_size )`.
     
     However, this does not affect a stalled motor.
     */
    real    unbinding_chance;
 
    /// probability to not unbind when attempting a step into the void (outside the fiber)
    /**
     The probability `dangling_chance` in `[0,1]` is tested if the Walker is at the last site
     on the Fiber, and attempts to step further. The result will be detachment with a probability
     `1 - dangling_chance`. This probability is per attempted step, the rate of which
     depends on the stepping rate `( max_speed / step_size )`.
     
     <em>default = 0</em>
     */
    real    dangling_chance;
    
    /// @}
    
private:
    
    /// derived variable
    real    var_rate_dt;
    
    /// derived variable
    real    stepping_rate, stepping_rate_dt;

    /// Indicates whether the walker moves to the plus or minus end
    bool    plusEndDirected;
public:

    /// constructor
    WalkerProp(const std::string& n) : DigitProp(n)  { clear(); }
    
    /// destructor
    ~WalkerProp() { }
    
    /// return a Hand with this property
    virtual Hand * newHand(HandMonitor* h) const;
    
    /// set default values
    void clear();
    
    /// set from a Glossary
    void read(Glossary&);
    
    /// compute values derived from the parameters
    void complete(Simul const*);
    
    /// perform additional tests for the validity of parameters, given the elasticity of the link
    void checkStiffness(real stiff, real len, real mul, real kT) const;
    
    /// return a carbon copy of object
    Property* clone() const { return new WalkerProp(*this); }

    /// write all values
    void write_values(std::ostream &) const;
    
};

#endif

