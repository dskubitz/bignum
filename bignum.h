#ifndef BIGNUM_H
#define BIGNUM_H

#include <stdexcept>
#include <vector>
#include <string>
#include <cassert>
#include <sstream>
#include <iostream>
#include <cstring>

template<class T, class U>
constexpr auto compare_3way(const T& a, const U& b)
{
	return (b < a) - (a < b);
}

constexpr auto bitmask(unsigned N)
{
	return ~(~0 << N);
}

constexpr unsigned log_2(unsigned v)
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

struct Magnitude : private std::basic_string<uint16_t> {
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
	using Vector = std::basic_string<uint16_t>;
	using size_type = typename Vector::size_type;
	using value_type = typename Vector::value_type;
	using Vector::swap;
	using Vector::size;
	using Vector::resize;
	using Vector::operator[];

	using int_type = long long;

	static constexpr unsigned long long Radix            = std::numeric_limits<value_type>::max() + 1;
	static constexpr unsigned           BitsPerDigit     = std::numeric_limits<value_type>::digits;
	static constexpr unsigned           Log2BitsPerDigit = log_2(BitsPerDigit);
	static constexpr auto               KaratsubaCutoff  = 20;

	template<class T>
	constexpr auto mod16(T val) { return val >> 4; }

	bool is_zero() const
	{
		return this->empty();
	}

	void strip_leading_zeroes()
	{
		while (!this->empty() && this->back() == 0)
			this->pop_back();
	}

	template<class U, class=std::enable_if_t<std::is_integral_v<U>>>
	static constexpr value_type value_cast(U val)
	{
		return static_cast<value_type>(val);
	}

	template<class U, class=std::enable_if_t<std::is_integral_v<U>>>
	static constexpr size_type size_cast(U val)
	{
		return static_cast<size_type>(val);
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude(U val)
	{
		convert_from(val);
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	Magnitude& operator=(U val)
	{
		convert_from(val);
		return *this;
	}

	Magnitude() = default;
	Magnitude(const Magnitude&) = default;
	Magnitude& operator=(const Magnitude&) = default;
	Magnitude(Magnitude&&) = default;
	Magnitude& operator=(Magnitude&&) = default;

	//@formatter:off
	explicit operator bool() { return !is_zero(); }
	explicit operator bool() const { return !is_zero(); }
	operator signed char() const { return convert_to<signed char>(); }
	operator unsigned char() const { return convert_to<unsigned char>(); }
	operator char() const { return convert_to<char>(); }
	operator wchar_t() const { return convert_to<wchar_t>(); }
	operator char16_t() const { return convert_to<char16_t>(); }
	operator char32_t() const { return convert_to<char32_t>(); }
	operator int() const { return convert_to<int>(); }
	operator unsigned int() const { return convert_to<unsigned int>(); }
	operator long int() const { return convert_to<long int>(); }
	operator unsigned long int() const { return convert_to<unsigned long int>(); }
	operator short int() const { return convert_to<short int>(); }
	operator unsigned short int() const { return convert_to<unsigned short int>(); }
	operator long long int() const { return convert_to<long long int>(); }
	operator unsigned long long int() const { return convert_to<unsigned long long int>(); }
	//@formatter:on

	template<typename T, T N>
	Magnitude(const char (& str)[N])
	{
		for (size_type i = 0; i < N - 1; ++i) {
			int c = str[i];
			if (isdigit(c)) {
				*this *= 10;
				*this += c - '0';
			} else if (isspace(c))
				continue;
			else
				throw MagnitudeError("bad input string");
		}
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

	int compare(const Magnitude& rhs) const
	{
		return compare(*this, rhs);
	}

	static int compare(const Magnitude& lhs, const Magnitude& rhs)
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
		*this += 1;
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
		*this -= 1;
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

	friend Magnitude square(const Magnitude& x)
	{
		/*
		 * Future note: optimize
		 */
		return x * x;
	}

	template<class U>
	U convert_to() const
	{
		if (is_zero())
			return 0;

		auto it = this->crbegin(), end = this->crend();

		unsigned long long res = *it++;

		for (; it != end; ++it) {
			unsigned long long i = *it;
			res = res * static_cast<unsigned long long>(Radix) + i;
		}
		return res;
	}

	void convert_from(bool b)
	{
		this->clear();
		if (b)
			this->push_back(1);
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	void convert_from(U val)
	{
		if (val < 0)
			underflow();
		this->clear();
		while (val) {
			this->push_back(value_cast(val % Radix));
			val /= Radix;
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

	friend bool operator<(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs.compare(rhs) == -1;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<(const Magnitude& lhs, U rhs)
	{
		return lhs.compare(rhs) == -1;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<(U lhs, const Magnitude& rhs)
	{
		return rhs.compare(lhs) == 1;
	}

	friend bool operator>(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs.compare(rhs) == 1;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>(const Magnitude& lhs, U rhs)
	{
		if (!rhs) return !lhs.is_zero();
		return lhs.compare(rhs) == 1;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>(U lhs, const Magnitude& rhs)
	{
		if (!lhs) return rhs.is_zero();
		return rhs.compare(lhs) == -1;
	}

	friend bool operator==(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs.compare(rhs) == 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator==(const Magnitude& lhs, U rhs)
	{
		if (!rhs) return lhs.is_zero();
		return lhs.compare(rhs) == 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator==(U lhs, const Magnitude& rhs)
	{
		if (!lhs) return rhs.is_zero();
		return rhs.compare(lhs) == 0;
	}

	friend bool operator!=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs.compare(rhs) != 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator!=(const Magnitude& lhs, U rhs)
	{
		if (!rhs) return !lhs.is_zero();
		return lhs.compare(rhs) != 0;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator!=(U lhs, const Magnitude& rhs)
	{
		if (!lhs) return !rhs.is_zero();
		return rhs.compare(lhs) != 0;
	}

	friend bool operator<=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs < rhs || lhs == rhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<=(const Magnitude& lhs, U rhs)
	{
		return lhs < rhs || lhs == rhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator<=(U lhs, const Magnitude& rhs)
	{
		return lhs < rhs || lhs == rhs;
	}

	friend bool operator>=(const Magnitude& lhs, const Magnitude& rhs)
	{
		return lhs > rhs || lhs == rhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>=(const Magnitude& lhs, U rhs)
	{
		if (!rhs) return true;
		return lhs > rhs || lhs == rhs;
	}

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
	friend bool operator>=(U lhs, const Magnitude& rhs)
	{
		return lhs > rhs || lhs == rhs;
	}

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
	friend Magnitude pow(const Magnitude& base, T exponent)
	{
		if (exponent < 0)
			throw MagnitudeError("negative exponent");
		if (!base)
			return !exponent ? 1 : 0;
		Magnitude result(1), base_to_pow2(base);
		while (exponent > 0) {
			if (exponent & 1)
				result *= base_to_pow2;
			if (exponent >>= 1) {
				base_to_pow2 = square(base_to_pow2);
			}
		}
		return result;
	}

	template<class T1, class T2, class = std::enable_if_t<std::is_integral_v<std::common_type_t<T1, T2>>>>
	friend Magnitude::int_type trial(const Magnitude& r, const Magnitude& d, T1 k, T2 m)
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
	friend bool smaller(const Magnitude& r, const Magnitude& dq, T1 k, T2 m)
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
	friend void sub_interval(Magnitude& r, const Magnitude& dq, T1 k, T2 m)
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

	friend std::pair<Magnitude, Magnitude> longdivide(const Magnitude& lhs, const Magnitude& rhs)
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
		d.resize(r.size() + 1);
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
	friend std::pair<Magnitude, Magnitude> divmod(const Magnitude& lhs, U rhs)
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

	friend std::pair<Magnitude, Magnitude> divmod(const Magnitude& lhs, const Magnitude& rhs)
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
	Bignum() = default;

	Bignum(const char* str) { parse(str); }

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
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
	Bignum(Bignum&&) = default;
	Bignum& operator=(const Bignum&) = default;
	Bignum& operator=(Bignum&&) = default;
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

	int signum() const { return signum_; }

	//@formatter:off
	explicit operator bool() const { return signum_; }
	operator signed char() const { return convert_to<signed char>(); }
	operator unsigned char() const { return convert_to<unsigned char>(); }
	operator char() const { return convert_to<char>(); }
	operator wchar_t() const { return convert_to<wchar_t>(); }
	operator char16_t() const { return convert_to<char16_t>(); }
	operator char32_t() const { return convert_to<char32_t>(); }
	operator int() const { return convert_to<int>(); }
	operator unsigned int() const { return convert_to<unsigned int>(); }
	operator long int() const { return convert_to<long int>(); }
	operator unsigned long int() const { return convert_to<unsigned long int>(); }
	operator short int() const { return convert_to<short int>(); }
	operator unsigned short int() const { return convert_to<unsigned short int>(); }
	operator long long int() const { return convert_to<long long int>(); }
	operator unsigned long long int() const { return convert_to<unsigned long long int>(); }
	//@formatter:on

	template<class T, class = std::enable_if_t<std::is_integral_v<T>>>
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

	/*
	 * absolute value
	 */
	friend Bignum abs(const Bignum& num)
	{
		Bignum res(num);
		res.signum_ = num.signum_ ? 1 : 0;
		return res;
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

private:
	bool is_negative() const { return signum_ < 0; }

	void normalize(bool negative_result)
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
		operator[](size_t index) const
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
	U convert_to() const
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

	template<class U, class = std::enable_if_t<std::is_integral_v<U>>>
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

#endif //BIGNUM_H
