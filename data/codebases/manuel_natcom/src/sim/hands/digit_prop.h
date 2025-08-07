// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#ifndef DIGIT_PROP_H
#define DIGIT_PROP_H

#include "hand_prop.h"


//#define DIGITS_GILLESPIE

/// Additional Property for Digit
/**
 @ingroup Properties
 */
class DigitProp : public HandProp
{
    friend class Digit;
    
public:
    
    /**
     @defgroup DigitPar Parameters of Digit
     @ingroup Parameters
     Inherits @ref HandPar.
     @{
     */
    
    /// distance between binding sites (should match fiber:lattice_unit)
    real    step_size;
    
    /// if true, obey the fiber:lattice (default=true)
    bool    use_lattice;
    
    /// @}
    
public:
    
    /// constructor
    DigitProp(const std::string& n) : HandProp(n)  {
        clear();
    }
    
    /// destructor
    ~DigitProp() { }
    
    /// return a Hand with this property
    virtual Hand * newHand(HandMonitor* h) const;
    
    /// set default values
    void clear();
    
    /// set from a Glossary
    void read(Glossary&);
    
    /// compute values derived from the parameters
    void complete(Simul const*);
    
    /// return a carbon copy of object
    Property* clone() const { return new DigitProp(*this); }
    
    /// write all values
    void write_values(std::ostream &) const;
    
#ifdef DIGITS_GILLESPIE
    real dt_for_gillespie;
#endif
    
#ifdef MULTI_LATTICE
    // Add to lattice_id to have both lat_val and multi lattice
    unsigned int lat_val_add;
    
    // Boolean indicating whether when a fiber approaches the attached hand can move to the right    lattice site so that it can bind to the fiber
    bool redistribute_on_binding;
#endif

    
    
    
};

#endif
