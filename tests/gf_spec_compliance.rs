//! GF(2^16) specification compliance tests.
//!
//! Validates Galois field arithmetic against the PAR2 specification and
//! exhaustively checks field properties that Reed-Solomon correctness
//! depends on.

use rust_par2::gf;

// =========================================================================
// PAR2 spec reference values
// =========================================================================

/// The PAR2 spec defines the irreducible polynomial as:
///   x^16 + x^12 + x^3 + x + 1 = 0x1100B
///
/// This means 2^16 = x^12 + x^3 + x + 1 = 0x100B in GF(2^16).
#[test]
fn test_generator_reduction() {
    // 2^16 mod polynomial should give 0x100B
    assert_eq!(gf::pow(2, 16), 0x100B);
    // 2^17 = 2 * 2^16 = 2 * 0x100B
    assert_eq!(gf::pow(2, 17), gf::mul(2, 0x100B));
}

/// Verify the first several powers of the generator (2) match known values.
/// These are derivable from the polynomial and serve as a cross-check.
#[test]
fn test_generator_powers() {
    assert_eq!(gf::exp2(0), 1);
    assert_eq!(gf::exp2(1), 2);
    assert_eq!(gf::exp2(2), 4);
    assert_eq!(gf::exp2(3), 8);
    assert_eq!(gf::exp2(4), 16);
    assert_eq!(gf::exp2(8), 256);
    assert_eq!(gf::exp2(15), 32768);
    assert_eq!(gf::exp2(16), 0x100B); // reduction by polynomial
}

/// The multiplicative group of GF(2^16) has order 65535.
/// The generator 2 must have exactly this order.
#[test]
fn test_generator_has_full_order() {
    // 2^65535 = 1
    assert_eq!(gf::pow(2, 65535), 1);

    // 2^k != 1 for all proper divisors of 65535
    // 65535 = 3 × 5 × 17 × 257
    // Proper divisors to check: 65535/3, 65535/5, 65535/17, 65535/257
    assert_ne!(gf::pow(2, 65535 / 3), 1, "order divides 65535/3");
    assert_ne!(gf::pow(2, 65535 / 5), 1, "order divides 65535/5");
    assert_ne!(gf::pow(2, 65535 / 17), 1, "order divides 65535/17");
    assert_ne!(gf::pow(2, 65535 / 257), 1, "order divides 65535/257");
}

// =========================================================================
// Field axiom verification
// =========================================================================

/// GF(2^16) addition is commutative: a + b = b + a
#[test]
fn test_addition_commutative_exhaustive() {
    // Test a representative sample (exhaustive over all 2^32 pairs is too slow)
    for a in (0..=65535u16).step_by(257) {
        for b in (0..=65535u16).step_by(251) {
            assert_eq!(gf::add(a, b), gf::add(b, a));
        }
    }
}

/// GF(2^16) addition is associative: (a + b) + c = a + (b + c)
#[test]
fn test_addition_associative() {
    let vals = [0u16, 1, 2, 255, 256, 1000, 32768, 65535, 0x100B, 0xAAAA];
    for &a in &vals {
        for &b in &vals {
            for &c in &vals {
                assert_eq!(
                    gf::add(gf::add(a, b), c),
                    gf::add(a, gf::add(b, c)),
                    "associativity failed for ({a}, {b}, {c})"
                );
            }
        }
    }
}

/// GF(2^16) multiplication is associative: (a * b) * c = a * (b * c)
#[test]
fn test_multiplication_associative() {
    let vals = [1u16, 2, 3, 42, 255, 256, 1000, 32768, 65535, 0x100B];
    for &a in &vals {
        for &b in &vals {
            for &c in &vals {
                assert_eq!(
                    gf::mul(gf::mul(a, b), c),
                    gf::mul(a, gf::mul(b, c)),
                    "associativity failed for ({a}, {b}, {c})"
                );
            }
        }
    }
}

/// Distributive law: a * (b + c) = a*b + a*c
#[test]
fn test_distributive_law() {
    let vals = [0u16, 1, 2, 3, 42, 255, 256, 1000, 32768, 65535, 0x100B];
    for &a in &vals {
        for &b in &vals {
            for &c in &vals {
                let lhs = gf::mul(a, gf::add(b, c));
                let rhs = gf::add(gf::mul(a, b), gf::mul(a, c));
                assert_eq!(lhs, rhs, "distributive failed for ({a}, {b}, {c})");
            }
        }
    }
}

/// Every non-zero element has a unique multiplicative inverse.
#[test]
fn test_all_nonzero_elements_have_inverses() {
    for a in 1..=65535u16 {
        let a_inv = gf::inv(a);
        assert_eq!(
            gf::mul(a, a_inv),
            1,
            "a={a}, inv(a)={a_inv}: product should be 1"
        );
        // Inverse of inverse is the original
        assert_eq!(gf::inv(a_inv), a, "inv(inv({a})) should be {a}");
    }
}

/// Multiplication by zero always gives zero.
#[test]
fn test_mul_zero_exhaustive() {
    for a in (0..=65535u16).step_by(1) {
        assert_eq!(gf::mul(a, 0), 0, "mul({a}, 0) should be 0");
        assert_eq!(gf::mul(0, a), 0, "mul(0, {a}) should be 0");
    }
}

/// a + a = 0 for all elements (characteristic 2).
#[test]
fn test_characteristic_two() {
    for a in (0..=65535u16).step_by(1) {
        assert_eq!(gf::add(a, a), 0, "a + a should be 0 in GF(2^16)");
    }
}

/// a - a = 0 and sub equals add in characteristic 2.
#[test]
fn test_sub_equals_add() {
    for a in (0..=65535u16).step_by(257) {
        for b in (0..=65535u16).step_by(251) {
            assert_eq!(gf::sub(a, b), gf::add(a, b));
        }
    }
}

/// pow(a, FIELD_ORDER - 1) = inv(a) for all non-zero a (Fermat's little theorem).
#[test]
fn test_pow_equals_inv() {
    // Test a sample since this is O(n * log(n)) per element
    let vals: Vec<u16> = (1..=65535u16).step_by(127).collect();
    for a in vals {
        assert_eq!(
            gf::pow(a, 65534),
            gf::inv(a),
            "pow({a}, 65534) should equal inv({a})"
        );
    }
}

/// exp2 wraps correctly at the field order boundary.
#[test]
fn test_exp2_wrapping() {
    for offset in 0..100u32 {
        assert_eq!(
            gf::exp2(offset),
            gf::exp2(65535 + offset),
            "exp2 should wrap at 65535"
        );
    }
}

/// div(a, b) = mul(a, inv(b)) for all non-zero b.
#[test]
fn test_div_equals_mul_inv() {
    let vals = [1u16, 2, 3, 42, 255, 1000, 32768, 65535, 0x100B];
    for &a in &vals {
        for &b in &vals {
            assert_eq!(
                gf::div(a, b),
                gf::mul(a, gf::inv(b)),
                "div({a}, {b}) should equal mul({a}, inv({b}))"
            );
        }
    }
}

// =========================================================================
// PAR2 input constants
// =========================================================================

/// Verify PAR2 input constants have full multiplicative order (65535).
/// This is required for the Vandermonde matrix to be non-singular.
#[test]
fn test_par2_input_constants_have_full_order() {
    let constants = rust_par2::matrix::par2_input_constants(100);

    for (i, &c) in constants.iter().enumerate() {
        assert_ne!(c, 0, "constant[{i}] must be non-zero");
        // c^65535 = 1
        assert_eq!(
            gf::pow(c, 65535),
            1,
            "constant[{i}]={c} must be in the field"
        );
        // c must have full order 65535 (not a subgroup element)
        assert_ne!(
            gf::pow(c, 65535 / 3),
            1,
            "constant[{i}]={c} has order dividing 65535/3"
        );
        assert_ne!(
            gf::pow(c, 65535 / 5),
            1,
            "constant[{i}]={c} has order dividing 65535/5"
        );
        assert_ne!(
            gf::pow(c, 65535 / 17),
            1,
            "constant[{i}]={c} has order dividing 65535/17"
        );
        assert_ne!(
            gf::pow(c, 65535 / 257),
            1,
            "constant[{i}]={c} has order dividing 65535/257"
        );
    }
}

/// All PAR2 input constants must be distinct.
#[test]
fn test_par2_input_constants_unique() {
    let constants = rust_par2::matrix::par2_input_constants(500);
    let mut seen = std::collections::HashSet::new();
    for (i, &c) in constants.iter().enumerate() {
        assert!(seen.insert(c), "duplicate constant at index {i}: {c}");
    }
}

/// The first PAR2 input constant corresponds to 2^1 (n=1 is the first valid exponent).
#[test]
fn test_par2_first_constants() {
    let constants = rust_par2::matrix::par2_input_constants(10);
    // Valid exponents skip multiples of 3, 5, 17, 257
    // n=1: valid (not divisible by 3,5,17,257) → c[0] = 2^1 = 2
    assert_eq!(constants[0], gf::exp2(1));
    // n=2: valid → c[1] = 2^2 = 4
    assert_eq!(constants[1], gf::exp2(2));
    // n=3: SKIP (multiple of 3)
    // n=4: valid → c[2] = 2^4 = 16
    assert_eq!(constants[2], gf::exp2(4));
    // n=5: SKIP (multiple of 5)
    // n=6: SKIP (multiple of 3)
    // n=7: valid → c[3] = 2^7 = 128
    assert_eq!(constants[3], gf::exp2(7));
    // n=8: valid → c[4] = 2^8 = 256
    assert_eq!(constants[4], gf::exp2(8));
}
