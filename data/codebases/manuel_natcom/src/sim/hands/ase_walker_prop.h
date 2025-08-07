// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef ASE_WALKER_PROP_H
#define ASE_WALKER_PROP_H

#include "digit_prop.h"

/// Additional Property for Walker
/**
 @ingroup Properties
*/
class AseWalkerProp : public DigitProp
{
    friend class AseWalker;
    
public:
    
    /// ADDED: diffusion rate on the fiber
    real    diffusion;
    
    real    dangling_chance;
    
    /// ADDED: boolean indicating wether the hand has rescuer activity
    //bool    is_rescuer;
    
    /// extra activity in case there is rescue
    real    rescue_prob;
    
    /// Probability of doing hold_shrinking_end
    real    hold_shrinking_prob;
    
#ifdef MULTI_LATTICE
    /// Rate of redistributing to a neighbouring lattice site
    real    redistribution_rate;
#endif
private:

    /// derived variable
    real    diff_rate_dt;
    
    /// derived variable
    real    diff2_rate_dt;
    
    /// derived variable
    real    step_size_kT;
    
    /// derived variable
    real    diff_rate;

    /// derived variable
    real    diff_rate_2;
    
    /// derived variable
#ifdef MULTI_LATTICE
    real    redistribution_rate_dt;
#endif
    /// taken from the couple, the delta G associated with a stretch of one step size divided by 2kT.
    real    U_step_kT_2;
    
    /// Same thing negatively exponentiated
    real    exp_U_step_kT_2;
    
public:

    /// constructor
    AseWalkerProp(const std::string& n) : DigitProp(n) { clear(); }
    
    /// destructor
    ~AseWalkerProp() { }
    
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
    Property* clone() const { return new AseWalkerProp(*this); }

    /// write all values
    void write_values(std::ostream &) const;

    void complete_from_couple(Simul const* sim, real stiffness);
};

#endif

