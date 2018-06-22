#ifndef BIGNUM_H
#define BIGNUM_H

#include <stdexcept>
#include <vector>
#include <string>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <tuple>
#include <numeric>

template<class T, class U>
constexpr auto compare_3way(const T& a, const U& b) noexcept
{
	return (b < a) - (a < b);
}

constexpr auto bitmask(unsigned N) noexcept
{
	return ~(~0 << N);
}

constexpr unsigned log_2(unsigned v) noexcept
{
	const unsigned b[] = {0x2, 0xc, 0xf0, 0xff00, 0xffff0000};
	const unsigned s[] = {1, 2, 4, 8, 16};
	unsigned       r   = 0;
	for (int       i   = sizeof(unsigned); i >= 0; --i) {
		if (v & b[i]) {
			v >>= s[i];
			r |= s[i];
		}
	}
	return r;
}

template<class Int, class = std::enable_if_t<std::is_integral_v<Int>>>
Int number_of_leading_zeroes(Int x) noexcept
{
	if (!x)
		return sizeof(x) * 8;
	Int n = 0;
	while (true) {
		if (x < 0) break;
		n++;
		x <<= 1;
	}
	return n;
}

struct Magnitude : private std::vector<uint16_t> {
private:
	struct MagnitudeError : std::logic_error {
		using std::logic_error::logic_error;
	};

	[[noreturn]] static void underflow(const std::string& msg = {})
	{
		if (!msg.empty())
			throw MagnitudeError("underflow -- " + msg);
		throw MagnitudeError("underflow");
	}

public:
	using Vector = std::vector<uint16_t>;
	using size_type = typename Vector::size_type;
	using value_type = typename Vector::value_type;
	using Vector::swap;
	using Vector::size;
	using Vector::resize;

	//using Vector::operator[];
	value_type& operator[](size_type idx)
	{
		return this->at(idx);
	}

	const value_type& operator[](size_type idx) const noexcept
	{
		const static value_type ZERO = 0;
		if (idx < size())
			return Vector::operator[](idx);
		else
			return ZERO;
	}

	using int_type = long long;

	static constexpr unsigned long long Radix            = std::numeric_limits<value_type>::max() + 1;
	static constexpr unsigned           BitsPerDigit     = std::numeric_limits<value_type>::digits;
	static constexpr unsigned           Log2BitsPerDigit = log_2(BitsPerDigit);
	static constexpr auto               KaratsubaCutoff  = 20;

	template<class T>
	constexpr auto mod16(T val) noexcept { return val >> 4; }

	bool is_zero() const noexcept
	{
		return this->empty();
	}

	void strip_leading_zeroes() noexcept
	{
		while (!this->empty() && this->back() == 0)
			this->pop_back();
	}

	template<class U>
	static constexpr value_type value_cast(U val) noexcept
	{
		return static_cast<value_type>(val);
	}

	template<class U>
	static constexpr size_type size_cast(U val) noexcept
	{
		return static_cast<size_type>(val);
	}

	template<class U, class = std::enable_if_t<std::is_arithmetic_v<U>>>
	Magnitude(U val)
	{
		convert_from(val);
	}

	template<class U, class = std::enable_if_t<std::is_arithmetic_v<U>>>
	Magnitude& operator=(U val)
	{
		convert_from(val);
		return *this;
	}

	Magnitude() noexcept = default;
	Magnitude(const Magnitude&) = default;
	Magnitude& operator=(const Magnitude&) = default;
	Magnitude(Magnitude&&) noexcept = default;
	Magnitude& operator=(Magnitude&&) noexcept = default;

	//@formatter:off
	explicit operator bool() const noexcept { return !is_zero(); }
	explicit operator signed char() const noexcept { return convert_to<signed char>(); }
	explicit operator unsigned char() const noexcept { return convert_to<unsigned char>(); }
	explicit operator char() const noexcept { return convert_to<char>(); }
	explicit operator wchar_t() const noexcept { return convert_to<wchar_t>(); }
	explicit operator char16_t() const noexcept { return convert_to<char16_t>(); }
	explicit operator char32_t() const noexcept { return convert_to<char32_t>(); }
	explicit operator int() const noexcept { return convert_to<int>(); }
	explicit operator unsigned int() const noexcept { return convert_to<unsigned int>(); }
	explicit operator long int() const noexcept { return convert_to<long int>(); }
	explicit operator unsigned long int() const noexcept { return convert_to<unsigned long int>(); }
	explicit operator short int() const noexcept { return convert_to<short int>(); }
	explicit operator unsigned short int() const noexcept { return convert_to<unsigned short int>(); }
	explicit operator long long int() const noexcept { return convert_to<long long int>(); }
	explicit operator unsigned long long int() const noexcept { return convert_to<unsigned long long int>(); }
	explicit operator float() const noexcept { return convert_to<float>(); }
	explicit operator double() const noexcept { return convert_to<double>(); }
	explicit operator long double() const noexcept { return convert_to<long double>(); }
	//@formatter:on

	size_type first_nonzero_word() const
	{
		auto sz = size();

		for (size_type i = 0; i < sz; ++i)
			if ((*this)[i])
				return i;

		throw MagnitudeError("first_nonzero_word() called on a zero magnitude");
	}

	int lowest_set_bit() const
	{
		int        offset = static_cast<int>(first_nonzero_word());
		value_type val    = (*this)[offset];
		int        pos    = 0;
		while ((val & 1) == 0) {
			val >>= 1;
			++pos;
		}
		return pos + offset * BitsPerDigit;
	}

	Magnitude(const char* str)
	{
		for (const char* p = str; *p; ++p) {
			int c = *p;
			if (isdigit(c)) {
				*this *= 10;
				*this += c - '0';
			} else if (isspace(c))
				continue;
			else
				throw MagnitudeError("bad input string");
		}
	}

	std::string str() const
	{
		return str_impl(*this);
	}

	int compare(const Magnitude& rhs) const noexcept
	{
		return compare(*this, rhs);
	}

	static int compare(const Magnitude& lhs, const Magnitude& rhs) noexcept
	{
		size_type lsz = lhs.size(),
			  rsz = rhs.size();
		if (lsz < rsz)
			return -1;
		if (lsz > rsz)
			return 1;
		if (lsz == 0)
			return 0;
		for (size_type i = lsz - 1; static_cast<int>(i) >= 0; --i)
			if (lhs[i] != rhs[i])
				return compare_3way(static_cast<long>(lhs[i]),
						    static_cast<long>(rhs[i]));
		return 0;
	}

	Magnitude& operator<<=(const Magnitude& rhs)
	{
		if (!rhs)
			return *this;
		if (rhs.size() <= 4) { // 64 bit value
			return *this <<= static_cast<unsigned long long>(rhs);
		} else {
			throw MagnitudeError("shift width too large");
		}
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator<<=(U rhs)
	{
		if (is_zero())
			return *this;
		if (!rhs)
			return *this;
		if (rhs < 0)
			throw MagnitudeError("negative shift amount");

		Magnitude& lhs = *this;
		unsigned long long n_digits = mod16(rhs);
		unsigned long long n_bits   = rhs & 0xfULL;
		size_type          sz       = size();

		if (n_bits == 0) {
			resize(sz + n_digits);
			std::move_backward(this->begin(), this->begin() + sz, this->end());
			std::fill(this->begin(), this->begin() + n_digits, 0);
		} else {
			unsigned n_bits2 = BitsPerDigit - n_bits;
			resize(sz + n_digits + 1);
			auto newMag = this->rbegin();
			auto mag    = this->rend() - sz;
			auto magEnd = this->rend();
			*newMag++ = lhs[sz - 1] >> n_bits2;
			while (mag != magEnd) {
				unsigned long val1 = static_cast<unsigned>(*mag++) << n_bits;
				unsigned long val2 = static_cast<unsigned>(*mag) >> n_bits2;
				*newMag++          = value_cast(val1 | val2);
			}
			while (newMag != magEnd) {
				*newMag++ = 0;
			}
		}

		strip_leading_zeroes();

		return *this;
	}

	Magnitude& operator>>=(const Magnitude& rhs)
	{
		if (!rhs)
			return *this;
		if (rhs.size() <= 4) { // 64 bit value
			return *this >>= static_cast<unsigned long long>(rhs);
		} else {
			throw MagnitudeError("shift width too large");
		}
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator>>=(U rhs)
	{
		if (is_zero())
			return *this;
		if (!rhs)
			return *this;
		if (rhs < 0)
			throw MagnitudeError("negative shift amount");

		Magnitude& lhs = *this;
		unsigned n_digits = rhs >> Log2BitsPerDigit;
		unsigned n_bits   = rhs & 0xfU;
		auto     sz       = size();

		if (n_digits >= sz) {
			convert_from(0);
			return *this;
		}

		if (n_digits != 0) {
			this->erase(this->begin(), this->begin() + n_digits);
		}

		if (n_bits != 0) {
			unsigned n_bits2 = 0x10 - n_bits;
			sz = size();
			auto high_bits = value_cast(static_cast<unsigned>(lhs[sz - 1]) >> n_bits);
			if (sz == 1) {
				lhs[0] = value_cast(static_cast<unsigned>(lhs[0]) >> n_bits);
			} else {
				for (int j  = sz - 1; j >= 1; --j) {
					lhs[j - 1] = value_cast(
						(static_cast<unsigned>(lhs[j]) << n_bits2)
						| (static_cast<unsigned>(lhs[j - 1]) >> n_bits));
				}
				lhs[sz - 1] = high_bits;
			}

		}
		strip_leading_zeroes();

		return *this;
	}

	Magnitude operator~() const
	{
		Magnitude res;
		auto      sz = size();
		res.resize(sz);
		for (size_type i = 0; i < sz; ++i) {
			res[i] = ~(*this)[i];
		}
		return res;
	}

	Magnitude& operator+=(const Magnitude& rhs)
	{
		Magnitude& lhs = *this;

		size_type llen  = lhs.size();
		size_type rlen  = rhs.size();
		size_type i;
		int_type  carry = 0, val;

		for (i = 0; i < llen; ++i) {
			val = i < rlen ? rhs[i] : 0;
			val += lhs[i] + carry;
			lhs[i] = value_cast(val % Radix);
			carry = val / Radix;
		}
		for (; i < rlen; ++i) {
			val = rhs[i] + carry;
			lhs.push_back(value_cast(val % Radix));
			carry = val / Radix;
		}
		while (carry) {
			lhs.push_back(value_cast(carry % Radix));
			carry /= Radix;
		}

		lhs.strip_leading_zeroes();
		return lhs;
	}

	template<typename U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator+=(U rhs)
	{
		Magnitude& lhs = *this;

		int_type  carry = 0, val;
		size_type sz    = lhs.size();

		for (size_type i = 0; i < sz; ++i) {
			val = lhs[i] + carry + (rhs % Radix);
			lhs[i] = value_cast(val % Radix);
			carry = val / Radix;
			rhs /= Radix;
		}
		while (rhs) {
			val = carry + (rhs % Radix);
			lhs.push_back(value_cast(val % Radix));
			carry = val / Radix;
			rhs /= Radix;
		}
		while (carry) {
			lhs.push_back(value_cast(carry % Radix));
			carry /= Radix;
		}

		lhs.strip_leading_zeroes();
		return lhs;
	}

	Magnitude operator++(int)
	{
		Magnitude old(*this);
		operator++();
		return old;
	}

	Magnitude operator++()
	{
		return *this += 1;
	}

	friend Magnitude operator+(Magnitude lhs, const Magnitude& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator+(Magnitude lhs, U rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator+(U lhs, Magnitude rhs)
	{
		rhs += lhs;
		return rhs;
	}

	friend Magnitude operator-(Magnitude lhs, const Magnitude& rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator-(Magnitude lhs, U rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator-(U lhs, const Magnitude& rhs)
	{
		Magnitude res(lhs);
		res -= rhs;
		return res;
	}

	Magnitude& operator-=(const Magnitude& rhs)
	{
		Magnitude& lhs = *this;
		size_type llen   = lhs.size();
		size_type rlen   = rhs.size();
		size_type i;
		int_type  borrow = 0, val, rval;

		if (llen < rlen)
			underflow();

		for (i = 0; i < llen; ++i) {
			rval = i < rlen ? rhs[i] : 0;
			val  = lhs[i] - rval - borrow + Radix;
			lhs[i] = value_cast(val % Radix);
			borrow = 1 - val / Radix;
		}

		lhs.strip_leading_zeroes();
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator-=(U rhs)
	{
		Magnitude& lhs = *this;
		size_type llen   = lhs.size();
		int_type  borrow = 0, val;

		for (size_type i = 0; i < llen; ++i) {
			val = lhs[i] - borrow - rhs % Radix + Radix;
			lhs[i] = value_cast(val % Radix);
			rhs /= Radix;
			borrow = 1 - val / Radix;
		}
		if (borrow != 0)
			throw MagnitudeError("underflow");

		lhs.strip_leading_zeroes();
		return lhs;
	}

	Magnitude operator--(int)
	{
		Magnitude old(*this);
		operator--();
		return old;
	}

	Magnitude& operator--()
	{
		return *this -= 1;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator*(Magnitude lhs, U rhs)
	{
		lhs *= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator*(U lhs, Magnitude rhs)
	{
		rhs *= lhs;
		return rhs;
	}

	friend Magnitude operator*(const Magnitude& lhs, const Magnitude& rhs)
	{
		size_type llen = lhs.size();
		size_type rlen = rhs.size();
		if (llen > KaratsubaCutoff && rlen > KaratsubaCutoff)
			return karatsuba(lhs, rhs);

		size_type i   = 0, j = 0;
		size_type val = 0, carry = 0;

		Magnitude res;
		res.resize(llen + rlen);

		for (i = 0; i < llen; ++i) {
			carry  = 0;
			for (j = 0; j < rlen; ++j) {
				val = static_cast<size_type>(lhs[i])
				      * static_cast<size_type>(rhs[j])
				      + carry
				      + static_cast<size_type>(res[i + j]);

				res[i + j] = value_cast(val % Radix);
				carry = val / Radix;
			}
			res[i + rlen] += value_cast(carry);
		}
		res.strip_leading_zeroes();
		return res;
	}

	Magnitude& operator*=(const Magnitude& rhs)
	{
		if (rhs.is_zero()) {
			convert_from(0);
			return *this;
		}

		Magnitude temp = *this * rhs;
		swap(temp);
		return *this;
	}

	template<typename U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator*=(U rhs)
	{
		if (!rhs) {
			convert_from(0);
			return *this;
		}

		Magnitude& lhs = *this;
		size_type llen  = lhs.size(), i;
		int_type  carry = 0, val;

		for (i = 0; i < llen; ++i) {
			val = lhs[i] * rhs + carry;
			lhs[i] = value_cast(val % Radix);
			carry = val / Radix;
		}
		while (carry) {
			lhs.push_back(value_cast(carry % Radix));
			carry /= Radix;
		}

		// This function is used in the longdivide implementation, and
		// by design does not strip leading zeroes, as the values used
		// there are padded with a leading zero and stripping it will
		// mess things up.
		// Note -- operator[] changed for const objects, the above comment
		// may no longer be the case.
		return lhs;
	}

	std::pair<Magnitude, Magnitude> split(size_type idx) const
	{
		Magnitude lower, upper;
		lower.assign(this->begin(), this->begin() + idx);
		upper.assign(this->begin() + idx, this->end());
		return {lower, upper};
	}

	static Magnitude karatsuba(const Magnitude& x, const Magnitude& y)
	{
		auto xlen = x.size();
		auto ylen = y.size();
		auto half = (std::max(xlen, ylen) + 1) / 2;

		auto&&[xl, xh] = x.split(half);
		auto&&[yl, yh] = y.split(half);

		Magnitude z0 = xl * yl;
		Magnitude z1 = (xh + xl) * (yh + yl);
		Magnitude z2 = xh * yh;

		return ((z2 << (BitsPerDigit * half)) + (z1 - z2 - z0) << BitsPerDigit * half) + z0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator/=(U rhs)
	{
		Magnitude& lhs = *this;
		size_type m     = lhs.size();
		int_type  carry = 0, val;

		for (size_type i = m - 1; static_cast<int>(i) >= 0; --i) {
			val = carry * Radix + lhs[i];
			lhs[i] = val / rhs;
			carry = val % rhs;
		}
		strip_leading_zeroes();
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator%=(U rhs)
	{
		Magnitude& lhs = *this;
		size_type m     = lhs.size();
		int_type  carry = 0, val;

		for (size_type i = m - 1; static_cast<int>(i) >= 0; --i) {
			val = carry * Radix + lhs[i];
			lhs[i] = val / rhs;
			carry = val % rhs;
		}
		convert_from(carry);
		strip_leading_zeroes();
		return *this;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator%(Magnitude lhs, U rhs)
	{
		lhs %= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator%(U lhs, const Magnitude& rhs)
	{
		Magnitude res(lhs);
		res %= rhs;
		return res;
	}

	friend Magnitude operator/(const Magnitude& lhs, const Magnitude& rhs)
	{
		return divmod(lhs, rhs).first;
	}

	friend Magnitude operator%(const Magnitude& lhs, const Magnitude& rhs)
	{
		return divmod(lhs, rhs).second;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator/(Magnitude lhs, U rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator/(U lhs, const Magnitude& rhs)
	{
		Magnitude res(lhs);
		res /= rhs;
		return res;
	}

	Magnitude& operator/=(const Magnitude& rhs)
	{
		Magnitude temp = *this / rhs;
		swap(temp);
		return *this;
	}

	Magnitude& operator%=(const Magnitude& rhs)
	{
		Magnitude temp = *this % rhs;
		swap(temp);
		return *this;
	}

	/*
	friend Magnitude square(const Magnitude& x)
	{
		return x * x;
	}
	*/

	template<class Int>
	Int convert_to() const noexcept
	{
		if (is_zero())
			return 0;

		auto it = this->crbegin(), end = this->crend();

		Int res = *it++;

		for (; it != end; ++it) {
			Int i = *it;
			res = res * static_cast<Int>(Radix) + i;
		}
		return res;
	}

	void convert_from(bool b)
	{
		this->clear();
		if (b)
			this->push_back(1);
	}

	template<class U, class = std::enable_if_t<std::is_arithmetic_v<U>>>
	void convert_from(U val)
	{
		convert_from(val, std::is_integral<U>{});
	}

	template<class Int>
	void convert_from(Int val, std::true_type)
	{
		if (val < 0)
			underflow();
		this->clear();
		while (val) {
			this->push_back(value_cast(val % Radix));
			val /= Radix;
		}
	}

	template<class Float>
	void convert_from(Float dval, std::false_type)
	{
		if (dval < 0)
			underflow();
		int    expo;
		double frac = frexp(dval, &expo);
		int    ndig = (expo - 1) / BitsPerDigit + 1;
		this->resize(ndig);
		frac = ldexp(frac, (expo - 1) % BitsPerDigit + 1);
		for (int i = ndig; --i >= 0;) {
			value_type bits = value_cast(frac);
			(*this)[i] = bits;
			frac = frac - (double) bits;
			frac = ldexp(frac, BitsPerDigit);
		}
	}

	static std::string str_impl(const Magnitude& val)
	{
		std::stringstream ss;
		auto&&[quo, rem] = divmod(val, 10);
		if (rem.is_zero()) ss << "0"; else ss << (int) rem;
		if (!quo.is_zero())
			return str_impl(quo) + ss.str();
		return ss.str();
	}

	friend Magnitude operator<<(Magnitude lhs, const Magnitude& rhs)
	{
		lhs <<= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator<<(Magnitude lhs, U rhs)
	{
		lhs <<= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator<<(U lhs, const Magnitude& rhs)
	{
		Magnitude res(lhs);
		res <<= rhs;
		return res;
	}

	friend Magnitude operator>>(Magnitude lhs, const Magnitude& rhs)
	{
		lhs >>= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator>>(Magnitude lhs, U rhs)
	{
		lhs >>= rhs;
		return lhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend Magnitude operator>>(U lhs, const Magnitude& rhs)
	{
		Magnitude res(lhs);
		res >>= rhs;
		return res;
	}

	// Comparison operators
	friend bool operator<(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) < 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<(const Magnitude& lhs, U rhs)
	{
		return compare(lhs, rhs) < 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<(U lhs, const Magnitude& rhs)
	{
		return compare(rhs, lhs) >= 0;
	}

	friend bool operator>(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) > 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>(const Magnitude& lhs, U rhs)
	{
		return compare(lhs, rhs) > 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>(U lhs, const Magnitude& rhs)
	{
		return rhs.compare(lhs) > 0;
	}

	friend bool operator==(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) == 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator==(const Magnitude& lhs, U rhs)
	{
		return lhs.compare(rhs) == 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator==(U lhs, const Magnitude& rhs)
	{
		return rhs.compare(lhs) == 0;
	}

	friend bool operator!=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) != 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator!=(const Magnitude& lhs, U rhs)
	{
		return compare(lhs, rhs) != 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator!=(U lhs, const Magnitude& rhs)
	{
		return compare(rhs, lhs) != 0;
	}

	friend bool operator<=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) <= 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<=(const Magnitude& lhs, U rhs)
	{
		return compare(lhs, rhs) <= 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<=(U lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) <= 0;
	}

	friend bool operator>=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) >= 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>=(const Magnitude& lhs, U rhs)
	{
		return compare(lhs, rhs) >= 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>=(U lhs, const Magnitude& rhs)
	{
		return compare(lhs, rhs) >= 0;
	}

	// Division implementation
	template<class T1, class T2, class = std::enable_if_t<std::is_integral_v<std::common_type_t<T1, T2>>>>
	static Magnitude::int_type trial(const Magnitude& r, const Magnitude& d, T1 k, T2 m)
	{
		using int_type = unsigned long long;
		constexpr auto Radix = Magnitude::Radix;
		int_type       km    = k + m;
		int_type       r3    = (r[km] * Radix + r[km - 1]) * Radix + r[km - 2];
		int_type       d2    = d[m - 1] * Radix + d[m - 2];
		int_type       left  = r3 / d2;
		int_type       right = Radix - 1;
		return left < right ? left : right;
	}

	template<class T1, class T2, class = std::enable_if_t<std::is_integral_v<std::common_type_t<T1, T2>>>>
	static bool smaller(const Magnitude& r, const Magnitude& dq, T1 k, T2 m)
	{
		using int_type = Magnitude::int_type;
		int_type i = m;
		int_type j = 0;

		while (i != j)
			if (r[i + k] != dq[i])
				j = i;
			else
				i -= 1;
		return r[i + k] < dq[i];
	}

	template<class T1, class T2, class = std::enable_if_t<std::is_integral_v<std::common_type_t<T1, T2>>>>
	static void sub_interval(Magnitude& r, const Magnitude& dq, T1 k, T2 m)
	{
		using size_type = Magnitude::size_type;
		constexpr auto Radix = Magnitude::Radix;

		long long borrow, diff;
		borrow = diff = 0;

		for (size_type i = 0; i <= m; ++i) {
			diff = static_cast<long long>(r[i + k])
			       - static_cast<long long>(dq[i])
			       - borrow + Radix;
			r[i + k] = Magnitude::value_cast(diff % Radix);
			borrow = 1 - diff / Radix;
		}

		if (borrow != 0)
			underflow();
	}

	static std::pair<Magnitude, Magnitude> longdivide(const Magnitude& lhs, const Magnitude& rhs)
	{
		using size_type = Magnitude::size_type;
		using int_type = Magnitude::int_type;

		size_type m = rhs.size();
		size_type n = lhs.size();
		size_type f = Magnitude::Radix / (rhs[m - 1] + 1);
		int_type  k = n - m;

		Magnitude r = lhs * f;
		r.resize(r.size() + 1);
		Magnitude d = rhs * f;
		d.resize(d.size() + 1);
		Magnitude q;
		q.resize(k + 1);
		Magnitude dq;
		dq.resize(k + 1);

		for (; k >= 0; --k) {
			long long qt = trial(r, d, k, m);
			dq = d * qt;
			if (smaller(r, dq, k, m)) {
				qt -= 1;
				dq = d * qt;
			}
			q[k] = Magnitude::value_cast(qt);
			sub_interval(r, dq, k, m);
		}
		r /= f;
		q.strip_leading_zeroes();
		r.strip_leading_zeroes();
		return {q, r};
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	static std::pair<Magnitude, Magnitude> divmod(const Magnitude& lhs, U rhs)
	{
		using size_type = Magnitude::size_type;
		using int_type = Magnitude::int_type;

		if (!rhs)
			throw MagnitudeError("divide by zero");

		size_type m = lhs.size();

		Magnitude x;
		x.resize(m);
		int_type carry = 0, val;

		for (int_type i = m - 1; i >= 0; --i) {
			val = carry * Magnitude::Radix + lhs[i];
			x[i] = val / rhs;
			carry = val % rhs;
		}
		x.strip_leading_zeroes();
		return {x, Magnitude(carry)};
	}

	static std::pair<Magnitude, Magnitude> divmod(const Magnitude& lhs, const Magnitude& rhs)
	{
		using size_type = Magnitude::size_type;
		using int_type = Magnitude::int_type;

		if (rhs.is_zero())
			throw MagnitudeError("divide by zero");

		size_type m = rhs.size();
		if (m == 1)
			return divmod(lhs, static_cast<unsigned>(rhs[m - 1]));

		int_type cmp = lhs.compare(rhs);
		if (cmp < 0)
			return {{}, lhs};
		if (cmp == 0)
			return {1, 0};
		return longdivide(lhs, rhs);
	}
};

struct BignumError : std::logic_error {
	using std::logic_error::logic_error;
};

class Bignum {
public:
	Bignum() noexcept = default;

	explicit Bignum(const char* str) { parse(str); }

	template<class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
	Bignum(T val)
	{
		if (val < 0) {
			signum_ = -1;
			mag     = Magnitude(-val);
		} else if (val > 0) {
			signum_ = 1;
			mag     = Magnitude(val);
		}
	}

	Bignum(const Bignum&) = default;
	Bignum(Bignum&&) noexcept = default;
	Bignum& operator=(const Bignum&) = default;
	Bignum& operator=(Bignum&&) noexcept = default;
	~Bignum() = default;

	void swap(Bignum& other) noexcept
	{
		using std::swap;
		swap(mag, other.mag);
		signum_ = signum_ ^ other.signum_;
		other.signum_ = other.signum_ ^ signum_;
		signum_ = signum_ ^ other.signum_;
	}

	std::string str() const
	{
		if (signum_ == -1)
			return "-" + mag.str();
		return mag.str();
	}

	int signum() const noexcept { return signum_; }

	//@formatter:off
	explicit operator bool() const noexcept { return signum_; }
	explicit operator signed char() const noexcept { return convert_to<signed char>(); }
	explicit operator unsigned char() const noexcept { return convert_to<unsigned char>(); }
	explicit operator char() const noexcept { return convert_to<char>(); }
	explicit operator wchar_t() const noexcept { return convert_to<wchar_t>(); }
	explicit operator char16_t() const noexcept { return convert_to<char16_t>(); }
	explicit operator char32_t() const noexcept { return convert_to<char32_t>(); }
	explicit operator int() const noexcept { return convert_to<int>(); }
	explicit operator unsigned int() const noexcept { return convert_to<unsigned int>(); }
	explicit operator long int() const noexcept { return convert_to<long int>(); }
	explicit operator unsigned long int() const noexcept { return convert_to<unsigned long int>(); }
	explicit operator short int() const noexcept { return convert_to<short int>(); }
	explicit operator unsigned short int() const noexcept { return convert_to<unsigned short int>(); }
	explicit operator long long int() const noexcept { return convert_to<long long int>(); }
	explicit operator unsigned long long int() const noexcept { return convert_to<unsigned long long int>(); }
	explicit operator float() const noexcept { return convert_to<float>(); }
	explicit operator double() const noexcept { return convert_to<double>(); }
	explicit operator long double() const noexcept { return convert_to<long double>(); }
	//@formatter:on

	template<class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
	Bignum& operator=(T val)
	{
		convert_from(val);
		return *this;
	}

	/*
	 * Unary plus and minus
	 */
	Bignum operator+() const
	{
		return *this;
	}

	Bignum operator-() const
	{
		Bignum copy(*this);
		copy.signum_ = -signum_;
		return copy;
	}

	/*
	 * Increment
	 */
	Bignum& operator++()
	{
		if (signum_ == -1) {
			--mag;
			if (mag.is_zero()) signum_ = 0;
			return *this;
		}
		if (signum_ == 0)
			signum_ = 1;
		++mag;
		return *this;
	}

	Bignum operator++(int)
	{
		Bignum tmp(*this);
		operator++();
		return tmp;
	}

	/*
	 * Decrement
	 */
	Bignum& operator--()
	{
		if (signum_ == 1) {
			--mag;
			if (mag.is_zero()) signum_ = 0;
			return *this;
		}
		if (signum_ == 0)
			signum_ = -1;
		++mag;
		return *this;
	}

	Bignum operator--(int)
	{
		Bignum tmp(*this);
		operator++();
		return tmp;
	}

	/*
	 * Equality
	 */
	friend bool operator==(const Bignum& lhs, const Bignum& rhs)
	{
		if (compare_3way(lhs.signum_, rhs.signum_))
			return false;
		return lhs.mag == rhs.mag;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator==(const Bignum& lhs, T rhs)
	{
		if (!rhs)
			return lhs.signum_ == 0;
		if (rhs > 0 && lhs.signum_ > 0)
			return lhs.mag == rhs;
		if (rhs < 0 && lhs.signum_ < 0)
			return lhs.mag == -rhs;
		return false;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator==(T lhs, const Bignum& rhs) { return rhs == lhs; }

	/*
	 * Inequality
	 */
	friend bool operator!=(const Bignum& lhs, const Bignum& rhs) { return !(rhs == lhs); }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator!=(T lhs, const Bignum& rhs) { return !(rhs == lhs); }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator!=(const Bignum& lhs, T rhs) { return !(lhs == rhs); }

	/*
	 * Less than
	 */
	friend bool operator<(const Bignum& lhs, const Bignum& rhs)
	{
		int cmp = compare_3way(lhs.signum_, rhs.signum_);
		if (!cmp) {
			if (lhs.signum_ == -1)
				return lhs.mag > rhs.mag;
			return lhs.mag < rhs.mag;
		} else return cmp != 1;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator<(const Bignum& lhs, T rhs)
	{
		if (!rhs)
			return lhs.signum_ < 0;
		if (rhs > 0)
			return lhs.signum_ > 0 ? lhs.mag < rhs : true;
		if (rhs < 0)
			return lhs.signum_ < 0 ? lhs.mag > -rhs : false;

		return false;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator<(T lhs, const Bignum& rhs) { return rhs >= lhs; }

	/*
	 * Greater than
	 */
	friend bool operator>(const Bignum& lhs, const Bignum& rhs) { return rhs < lhs; }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator>(const Bignum& lhs, T rhs)
	{
		if (!rhs)
			return lhs.signum_ > 0;
		if (rhs > 0)
			return lhs.signum_ > 0 ? lhs.mag > rhs : false;
		if (rhs < 0)
			return lhs.signum_ < 0 ? lhs.mag < -rhs : true;

		return false;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator>(T lhs, const Bignum& rhs)
	{
		return rhs <= lhs;
	}

	/*
	 * Less than or equal
	 */
	friend bool operator<=(const Bignum& lhs, const Bignum& rhs) { return !(rhs < lhs); }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator<=(const Bignum& lhs, T rhs)
	{
		if (!rhs)
			return lhs.signum_ <= 0;
		if (rhs > 0)
			return lhs.signum_ > 0 ? lhs.mag <= rhs : true;
		if (rhs < 0)
			return lhs.signum_ < 0 ? lhs.mag >= -rhs : false;

		return false;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator<=(T lhs, const Bignum& rhs) { return rhs > lhs; }

	/*
	 * Greater than or equal
	 */
	friend bool operator>=(const Bignum& lhs, const Bignum& rhs) { return !(lhs < rhs); }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator>=(const Bignum& lhs, T rhs)
	{
		if (!rhs)
			return lhs.signum_ >= 0;
		if (rhs > 0)
			return lhs.signum_ > 0 ? lhs.mag >= rhs : false;
		if (rhs < 0)
			return lhs.signum_ < 0 ? lhs.mag <= -rhs : true;

		return false;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend bool operator>=(T lhs, const Bignum& rhs) { return rhs < lhs; }

	/*
	 * Addition
	 */
	Bignum& operator+=(const Bignum& rhs)
	{
		if (!rhs) {
			return *this;
		} else if (signum_ == 0) {
			signum_ = rhs.signum_;
			mag     = rhs.mag;
			return *this;
		}

		if (signum_ == rhs.signum_) {
			mag += rhs.mag;
			return *this;
		}

		int cmp = mag.compare(rhs.mag);
		if (cmp == 0) {
			convert_from(0);
			return *this;
		}

		if (cmp < 0)
			mag = rhs.mag - mag;
		else
			mag -= rhs.mag;

		signum_ = cmp == signum_ ? 1 : -1;

		return *this;
	}

	friend Bignum operator+(Bignum lhs, const Bignum& rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator+(Bignum lhs, T rhs)
	{
		lhs += rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator+(T lhs, Bignum rhs)
	{
		rhs += lhs;
		return rhs;
	}

	/*
	 * Subtraction
	 */
	Bignum& operator-=(const Bignum& rhs)
	{
		if (!rhs)
			return *this;
		if (signum_ == 0) {
			signum_ = -rhs.signum_;
			mag     = rhs.mag;
			return *this;
		}

		if (signum_ != rhs.signum_) {
			mag += rhs.mag;
			return *this;
		}

		int cmp = mag.compare(rhs.mag);
		if (cmp == 0) {
			convert_from(0);
			return *this;
		}
		if (cmp > 0)
			mag -= rhs.mag;
		else
			mag = rhs.mag - mag;

		signum_ = cmp == signum_ ? 1 : -1;

		return *this;
	}

	friend Bignum operator-(Bignum lhs, const Bignum& rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator-(Bignum lhs, T rhs)
	{
		lhs -= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator-(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res -= rhs;
		return res;
	}

	/*
	 * Multiplication
	 */
	Bignum& operator*=(const Bignum& rhs)
	{
		if (signum_ == 0)
			return *this;
		if (rhs.signum_ == 0) {
			convert_from(0);
			return *this;
		}
		mag *= rhs.mag;
		signum_ = signum_ == rhs.signum_ ? 1 : -1;
		return *this;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	Bignum& operator*=(T rhs)
	{
		if (signum_ == 0)
			return *this;
		if (!rhs) {
			convert_from(0);
			return *this;
		}
		if (rhs < 0) {
			signum_ = -signum_;
			rhs     = -rhs;
		}
		mag *= rhs;
		return *this;
	}

	friend Bignum operator*(Bignum lhs, const Bignum& rhs)
	{
		lhs *= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator*(Bignum lhs, T rhs)
	{
		lhs *= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator*(T lhs, Bignum rhs)
	{
		rhs *= lhs;
		return rhs;
	}

	/*
	 * Division
	 */
	Bignum& operator/=(const Bignum& rhs)
	{
		// Magnitude class will catch divide by zero errors
		if (this == &rhs) {
			convert_from(1);
			return *this;
		}
		if (signum_ == 0 && rhs)  // if rhs is 0 an error will be raised later
			return *this;

		mag /= rhs.mag;
		if (mag.is_zero())
			signum_ = 0;
		else
			signum_ = signum_ == rhs.signum_ ? 1 : -1;
		return *this;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	Bignum& operator/=(T rhs)
	{
		if (signum_ == 0 && rhs)
			return *this;
		if (rhs < 0) {
			signum_ = -signum_;
			rhs     = -rhs;
		}
		mag /= rhs;
		if (mag.is_zero())
			signum_ = 0;
		return *this;
	}

	friend Bignum operator/(Bignum lhs, const Bignum& rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator/(Bignum lhs, T rhs)
	{
		lhs /= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator/(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res /= rhs;
		return res;
	}

	/*
	 * Remainder
	 */
	Bignum& operator%=(const Bignum& rhs)
	{
		// Magnitude class will catch divide by zero errors
		if (this == &rhs) {
			convert_from(1);
			return *this;
		}
		if (signum_ == 0 && rhs)  // if rhs is 0 an error will be raised later
			return *this;

		mag %= rhs.mag;
		if (mag.is_zero())
			signum_ = 0;
		return *this;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	Bignum& operator%=(T rhs)
	{
		if (signum_ == 0 && rhs)
			return *this;
		if (rhs < 0) {
			rhs = -rhs;
		}
		mag %= rhs;
		if (mag.is_zero())
			signum_ = 0;
		return *this;
	}

	friend Bignum operator%(Bignum lhs, const Bignum& rhs)
	{
		lhs %= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator%(Bignum lhs, T rhs)
	{
		lhs %= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator%(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res %= rhs;
		return res;
	}

	friend std::pair<Bignum, Bignum> divmod(Bignum n, Bignum d)
	{
		auto&&[qmag, rmag] = Magnitude::divmod(n.mag, d.mag);
		Bignum q(std::move(qmag), n.signum_ == d.signum_ ? 1 : -1); // constructor will handle zero case
		Bignum r(std::move(rmag), n.signum_);
		return {q, r};
	}

	/*
	 * Left shift
	 */
	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	Bignum& operator<<=(T rhs)
	{
		mag <<= rhs;
		if (mag.is_zero()) signum_ = 0;
		return *this;
	}

	Bignum& operator<<=(const Bignum& rhs)
	{
		if (rhs.signum_ < 0)
			throw BignumError("negative shift amount");
		if (sizeof(Magnitude::value_type) * rhs.mag.size() <= sizeof(unsigned long long)) {
			return *this <<= static_cast<unsigned long long>(rhs);
		} else throw BignumError("shift width too large");
	}

	friend Bignum operator<<(Bignum lhs, const Bignum& rhs)
	{
		lhs <<= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator<<(Bignum lhs, T rhs)
	{
		lhs <<= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator<<(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res <<= rhs;
		return res;
	}

	/*
	 * Right shift
	 */
	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	Bignum& operator>>=(T rhs)
	{
		bool onesLost = false;
		if (signum_ < 0) {
			unsigned n_digits = rhs >> Magnitude::Log2BitsPerDigit;
			unsigned n_bits   = rhs & bitmask(Magnitude::Log2BitsPerDigit);

			for (unsigned i = 0; i < n_digits; ++i)
				if ((onesLost = mag[i] != 0))
					break;

			if (!onesLost && n_bits != 0 && (mag[n_digits] & bitmask(n_bits)))
				onesLost = true;
		}
		mag >>= rhs;

		if (mag.is_zero()) {
			if (signum_ < 0) {
				convert_from(-1);
				return *this;
			}
			signum_ = 0;
		}
		if (onesLost)
			mag += 1;

		return *this;
	}

	Bignum& operator>>=(const Bignum& rhs)
	{
		if (rhs.signum_ < 0)
			throw BignumError("negative shift amount");
		if (sizeof(Magnitude::value_type) * rhs.mag.size() <= sizeof(unsigned long long)) {
			return *this >>= static_cast<unsigned long long>(rhs);
		} else throw BignumError("shift width too large");
	}

	friend Bignum operator>>(Bignum lhs, const Bignum& rhs)
	{
		lhs >>= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator>>(Bignum lhs, T rhs)
	{
		lhs >>= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator>>(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res >>= rhs;
		return res;
	}

	/*
	 * And
	 */
	Bignum& operator&=(const Bignum& rhs)
	{
		if (this == &rhs)
			return *this;
		size_t len = std::max(mag.size(), rhs.mag.size());
		mag.resize(len);

		sign_extended x(mag, signum_ < 0);
		sign_extended y(rhs.mag, rhs.signum_ < 0);

		for (size_t i = 0; i < len; ++i)
			mag[i] = x[i] & y[i];

		normalize(is_negative() && rhs.is_negative());
		return *this;
	}

	friend Bignum operator&(Bignum lhs, const Bignum& rhs)
	{
		lhs &= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator&(Bignum lhs, T rhs)
	{
		lhs &= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator&(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res &= rhs;
		return res;
	}

	/*
	 * Or
	 */
	Bignum& operator|=(const Bignum& rhs)
	{
		if (this == &rhs)
			return *this;
		size_t len = std::max(mag.size(), rhs.mag.size());
		mag.resize(len);

		sign_extended x(mag, signum_ < 0);
		sign_extended y(rhs.mag, rhs.signum_ < 0);

		for (size_t i = 0; i < len; ++i)
			mag[i] = x[i] | y[i];

		normalize(is_negative() || rhs.is_negative());
		return *this;
	}

	friend Bignum operator|(Bignum lhs, const Bignum& rhs)
	{
		lhs |= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator|(Bignum lhs, T rhs)
	{
		lhs |= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator|(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res |= rhs;
		return res;
	}

	/*
	 * Xor
	 */
	Bignum& operator^=(const Bignum& rhs)
	{
		if (this == &rhs) {
			convert_from(0);
			return *this;
		}
		size_t len = std::max(mag.size(), rhs.mag.size());
		mag.resize(len);

		sign_extended x(mag, signum_ < 0);
		sign_extended y(rhs.mag, rhs.signum_ < 0);

		for (size_t i = 0; i < len; ++i)
			mag[i] = x[i] ^ y[i];

		normalize((is_negative() || rhs.is_negative())
			  && (!is_negative() || !rhs.is_negative()));
		return *this;
	}

	friend Bignum operator^(Bignum lhs, const Bignum& rhs)
	{
		lhs ^= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator^(Bignum lhs, T rhs)
	{
		lhs ^= rhs;
		return lhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Bignum operator^(T lhs, const Bignum& rhs)
	{
		Bignum res(lhs);
		res ^= rhs;
		return res;
	}

	/*
	 * Complement
	 */
	Bignum operator~() const
	{
		Bignum res;
		res.mag.resize(mag.size());
		res.signum_ = signum_;
		size_t        sz = mag.size(), i;
		sign_extended x(mag, signum_ < 0);
		for (i = 0; i < sz; ++i) {
			res.mag[i] = ~x[i];
		}
		res.normalize(!res.is_negative());
		return res;
	}

	friend std::ostream& operator<<(std::ostream& os, const Bignum& num)
	{
		return os << num.str();
	}

	friend Bignum gcd(Bignum a, Bignum b)
	{
		if (!a) return b;
		if (!b) return a;
		if (a < 0) a = -a;
		if (b < 0) b = -b;
		if (a.mag.size() <= 4 && b.mag.size() <= 4)
			return Bignum(binary_gcd(static_cast<unsigned long long>(a),
						 static_cast<unsigned long long>(b)));
		Bignum t;
		while (b != 0) {
			if (abs(a.mag.size() - b.mag.size()) < 2)
				return binary_gcd(a, b);
			t = a % b;
			a = b;
			b = t;
		}
		return a;
	}

private:
	Bignum(Magnitude&& m, int signum)
		: mag(std::move(m)), signum_(static_cast<int8_t>(signum))
	{
		if (mag.is_zero()) signum_ = 0;
	}

	static Bignum binary_gcd(Bignum u, Bignum v)
	{
		int shift = std::min(u.mag.lowest_set_bit(), v.mag.lowest_set_bit());
		u >>= shift;
		v >>= shift;

		u >>= u.mag.lowest_set_bit();

		do {
			if (u.mag.size() <= 4 && v.mag.size() <= 4)
				return Bignum(binary_gcd(static_cast<unsigned long long>(u),
							 static_cast<unsigned long long>(v))) << shift;

			v >>= v.mag.lowest_set_bit();
			if (u > v)
				u.swap(v);
			v -= u;
		} while (v != 0);

		return u << shift;
	}

	template<class Int1, class Int2,
		 class = std::enable_if_t<std::is_integral_v<std::common_type_t<Int1, Int2>>>>
	static std::common_type_t<Int1, Int2> binary_gcd(Int1 u, Int2 v) noexcept
	{
		using ResultType = std::common_type_t<Int1, Int2>;
		ResultType shift = 0;

		while (((u | v) & 1) == 0) {
			u >>= 1;
			v >>= 1;
			++shift;
		}

		while ((u & 1) == 0)
			u >>= 1;

		do {
			while ((v & 1) == 0)
				v >>= 1;

			if (u > v) {
				u = u ^ v;
				v = v ^ u;
				u = u ^ v;
			}

			v -= u;
		} while (v != 0);

		return u << shift;
	}

	bool is_negative() const noexcept { return signum_ < 0; }

	void normalize(bool negative_result) noexcept
	{
		mag.strip_leading_zeroes();
		if (mag.is_zero()) {
			signum_ = 0;
		} else if (negative_result) {
			signum_ = -1;
			mag     = ~mag + 1;
		} else {
			signum_ = 1;
		}
	}

	struct sign_extended {
		sign_extended(const Magnitude& mag, bool is_neg = false)
			: mag_(mag), first_nonzero_short(0), is_negative(is_neg)
		{
			if (is_negative)
				for (unsigned i = 0; i < mag.size(); ++i) {
					if (mag[i]) {
						first_nonzero_short = i;
						break;
					}
				}
		}

		std::make_signed_t<Magnitude::value_type>
		operator[](size_t index) const noexcept
		{
			if (index < 0)
				return 0;
			if (index >= mag_.size())
				return is_negative ? -1 : 0;

			int res = mag_[index];
			if (!is_negative)
				return res;
			return index <= first_nonzero_short ? -res : ~res;
		}

		const Magnitude& mag_;
		unsigned first_nonzero_short;
		bool     is_negative;
	};

	void parse(const char* str)
	{
		int c;
		while (*str) {
			c = *str++;
			if (isdigit(c)) {
				--str;
				mag = Magnitude(str);

				if (mag && !signum_) signum_ = 1;
				return;
			}
			if (c == '+') {
				signum_ = 1;
			} else if (c == '-') {
				signum_ = -1;
			} else {
				throw BignumError("invalid input string");
			}
		}
	}

	template<class U>
	U convert_to() const noexcept
	{
		if (!mag)
			return 0;
		U res = mag.convert_to<U>();
		if (signum_ == -1)
			res = -res;

		return res;
	}

	void convert_from(bool b)
	{
		signum_ = b;
		mag.convert_from(b);
	}

	template<class U>
	void convert_from(U val)
	{
		if (!val) {
			signum_ = 0;
			mag.convert_from(0);
		} else if (val < 0) {
			signum_ = -1;
			mag.convert_from(-val);
		} else {
			signum_ = 1;
			mag.convert_from(val);
		}
	}

	Magnitude mag;
	int8_t    signum_{0};
};

inline Bignum abs(Bignum x)
{
	return x < 0 ? -x : x;
}

inline std::tuple<Bignum, Bignum> as_integer_ratio(double x)
{
	int    exponent;
	double float_part = frexp(x, &exponent);
	std::cout << float_part << ' ' << exponent << '\n';
	for (int i = 0; i < 300 && float_part != floor(float_part); ++i) {
		float_part *= 2.0;
		--exponent;
	}
	return {Bignum(float_part), Bignum(1) << abs(exponent)};
}

#endif //BIGNUM_H
