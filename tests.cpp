#include "bignum.h"
#include <random>
#include <gtest/gtest.h>

template<class T>
void printbits(T num)
{
	unsigned      nbits = sizeof(T) * 8;
	char          bits[nbits];
	for (unsigned i     = 0; i < nbits; ++i) {
		bits[i] = (num & 1) + '0';
		num >>= 1;
	}
	for (unsigned i     = 0; i < nbits; ++i) {
		putchar(bits[nbits - i - 1]);
	}
	putchar('\n');
}

TEST(Magnitude, FirstSetBit)
{
	Magnitude m("36893488147419103232");
	EXPECT_EQ(m.lowest_set_bit(), 65);
}

#define test_binop(op) \
[&](const auto& l, const auto& r) {\
        Bignum bl = l; \
        Bignum br = r; \
        auto result = l op r; \
        EXPECT_EQ(bl op br, result) << l << #op << r; \
        EXPECT_EQ(bl op r, result) << "mixed mode failed for right hand integer-type"; \
        EXPECT_EQ(l op br, result) << "mixed mode failed for left hand integer-type"; \
}

#define test_unary(op) \
[&](const auto& r) {\
        Bignum br = r; \
        EXPECT_EQ(op br, op r) << #op << r; \
}

struct Rand2 {
	std::random_device                 rd;
	std::mt19937                       gen;
	std::uniform_int_distribution<int> dis;

	Rand2()
		: rd{}, gen{rd()}, dis(std::numeric_limits<int>::min() / 2, std::numeric_limits<int>::max() / 2) { }

	auto operator()() { return dis(gen); }
};

TEST(Bignum, CountLeadingZeroes)
{
	Rand2 rand;
	int   trials = 100;
	while (trials--) {
		auto val = rand();
		ASSERT_EQ(number_of_leading_zeroes(val), __builtin_clz(val));
	}
}

TEST(Bignum, Less)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(<);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Greater)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(>);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, LessEqual)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(<=);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, GreaterEqual)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(>=);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Equal)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(==);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, NotEqual)
{
	Rand2 rand;
	int   trials = 1000;
	auto  func   = test_binop(!=);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Addition)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(+);
	while (trials--) {
		auto l = rand(), r = rand();
		func(l, r);
	}
}

TEST(Bignum, Subtraction)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(-);
	while (trials--) {
		func(rand(), rand());
	}
}

TEST(Bignum, Multiplication)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(*);
	while (trials--) {
		auto l = rand() & 0x7fff, r = rand() % 0x7fff;
		func(l, r);
	}
}

/*
TEST(Bignum, Division)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(/);
	while (trials--) {
		auto l = rand() % 655360, r = rand() % 10000;
		if (!r) {
			++trials;
			continue;
		}
		func(l, r);
	}
}
*/

TEST(Bignum, Remainder)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(%);
	while (trials--) {
		auto l = rand(), r = rand();
		if (!r) {
			++trials;
			continue;
		}
		func(l, r);
	}
}

TEST(Bignum, LeftShift)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(<<);
	while (trials--) {
		func(rand() % 65536, std::abs(rand()) % 15);
	}
}

TEST(Bignum, RightShift)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(>>);
	while (trials--) {
		func(rand(), std::abs(rand()) % 15);
	}
}

TEST(Bignum, And)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(&);
	ASSERT_EQ(Bignum(43883) & 21652, 43883 & 21652);
	ASSERT_EQ(Bignum(855696157) & Bignum(-937842771), 855696157 & -937842771);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Or)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(|);
	ASSERT_EQ(Bignum(43883) | 21652, 43883 | 21652);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Xor)
{
	Rand2  rand;
	Bignum a, b;
	int    trials = 1000;
	auto   func   = test_binop(^);
	ASSERT_EQ(Bignum(43883) ^ 21652, 43883 ^ 21652);
	while (trials--)
		func(rand(), rand());
}

TEST(Bignum, Compl)
{
	Rand2  rand;
	Bignum a;
	int    trials = 1000;
	ASSERT_EQ(~Bignum(43883), -43884);
	auto   func   = test_unary(~);
	while (trials--)
		func(rand());
}

TEST(Bignum, Constructor)
{
	Bignum       b("12345");
	EXPECT_EQ(b.str(), "12345");
	EXPECT_EQ(int(b), 12345);
	Bignum       b1("-123");
	EXPECT_EQ(b1.str(), "-123");
	EXPECT_EQ(int(b1), -123);
	Bignum       b2(12345);
	EXPECT_EQ(int(b2), 12345);
	Bignum       b3(-12345);
	EXPECT_EQ(int(b3), -12345);
}

TEST(Bignum, IncrementDecrement)
{
	Bignum i = 0;
	ASSERT_EQ(i.signum(), 0);
	++i;
	ASSERT_EQ(i.signum(), 1);
	ASSERT_EQ(i, 1);
	++i;
	ASSERT_EQ(i, 2);
	--i;
	ASSERT_EQ(i.signum(), 1);
	ASSERT_EQ(i, 1);
	--i;
	ASSERT_EQ(i.signum(), 0);
	ASSERT_EQ(i, 0);
	--i;
	ASSERT_EQ(i.signum(), -1);
	ASSERT_EQ(i, -1);
	++i;
	ASSERT_EQ(i.signum(), 0);
	ASSERT_EQ(i, 0);
}

Bignum fact(Bignum x)
{
	if (x <= 1)
		return 1;
	return x * fact(x - 1);
}

TEST(Bignum, Factorial)
{
	EXPECT_EQ(fact(0).str(), "1");
	EXPECT_EQ(fact(1).str(), "1");
	EXPECT_EQ(fact(2).str(), "2");
	EXPECT_EQ(fact(3).str(), "6");
	EXPECT_EQ(fact(4).str(), "24");
	EXPECT_EQ(fact(5).str(), "120");
	EXPECT_EQ(fact(6).str(), "720");
	EXPECT_EQ(fact(7).str(), "5040");
	EXPECT_EQ(fact(8).str(), "40320");
	EXPECT_EQ(fact(9).str(), "362880");
	EXPECT_EQ(fact(10).str(), "3628800");
	EXPECT_EQ(fact(11).str(), "39916800");
	EXPECT_EQ(fact(12).str(), "479001600");
	EXPECT_EQ(fact(13).str(), "6227020800");
	EXPECT_EQ(fact(14).str(), "87178291200");
	EXPECT_EQ(fact(15).str(), "1307674368000");
	EXPECT_EQ(fact(16).str(), "20922789888000");
	EXPECT_EQ(fact(17).str(), "355687428096000");
	EXPECT_EQ(fact(18).str(), "6402373705728000");
	EXPECT_EQ(fact(19).str(), "121645100408832000");
	EXPECT_EQ(fact(19).str(), "121645100408832000");
	EXPECT_EQ(fact(20).str(), "2432902008176640000");
	EXPECT_EQ(fact(35).str(), "10333147966386144929666651337523200000000");
	EXPECT_EQ(fact(100).str(),
		  "93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000");
}

TEST(Bignum, ConversionToFloating)
{
	double d     = 42;
	Bignum b     = 42;
	double d2    = static_cast<double>(b);
	EXPECT_EQ(d, d2);
	double dval  = 3.14159e15;
	double dval2 = -3.14159e15;
	Bignum expect("3141590000000000");
	Bignum bval  = dval;
	Bignum bval2 = dval2;
	EXPECT_EQ(bval, expect);
	EXPECT_EQ(bval2, -expect);
}

TEST(Bignum, GCD)
{
	Rand2 rand;
	int   trials = 100;
	while (trials--) {
		auto a = rand(), b = rand();
		if (a || b) {
			Bignum ba(a), bb(b);
			EXPECT_EQ(gcd(ba, bb), std::gcd(a, b));
		} else {
			trials++;
		}
	}
}

TEST(Bignum, Divmod)
{
	Rand2 rand;
	int trials = 100;
	while (trials--) {
		auto a = rand(), b = rand();
		if (b) {
			Bignum ba(a), bb(b);
			auto&&[q, r] = divmod(ba, bb);
			EXPECT_EQ(q, a/b);
			EXPECT_EQ(r, a%b);
		} else {
			trials++;
		}
	}
}