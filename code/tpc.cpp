#include <bits/stdc++.h>
using namespace std;
#define rep(i, s, t) for (auto i = (s); i < (t); i++)
#define repb(i, s, t) for (auto i = (s); i <= (t); i++)
#define dd(x) cout << #x << '=' << x << ' '
#define de(x) cout << #x << '=' << x << '\n'

typedef vector<double> vd;
typedef double (*uf)(double);
typedef double (*af)(double, double);

random_device rd;

vd operator*(vd a, vd b)
{
    assert(a.size() == b.size());
    vd c(a.size());
    rep(i, 0, a.size()) c[i] = a[i] * b[i];
    return c;
}

vd operator*(double a, vd b)
{
    vd c(b.size());
    rep(i, 0, b.size()) c[i] = a * b[i];
    return c;
}

vd operator/(vd a, vd b)
{
    assert(a.size() == b.size());
    vd c(a.size());
    rep(i, 0, a.size()) c[i] = a[i] / b[i];
    return c;
}

vd operator/(vd a, double b)
{
    vd c(a.size());
    rep(i, 0, a.size()) c[i] = a[i] / b;
    return c;
}

vd operator+(vd a, vd b)
{
    assert(a.size() == b.size());
    vd c(a.size());
    rep(i, 0, a.size()) c[i] = a[i] + b[i];
    return c;
}

vd operator-(vd a, vd b)
{
    assert(a.size() == b.size());
    vd c(a.size());
    rep(i, 0, a.size()) c[i] = a[i] - b[i];
    return c;
}

template <class T>
vd ufunc(vd a, T f)
{
    vd b(a.size());
    rep(i, 0, a.size()) b[i] = f(a[i]);
    return b;
}

vd urd(size_t n)
{
    vd a(n);
    rep(i, 0, a.size()) a[i] = rd();
    return a / afunc(a, plus());
}

template <class T>
double afunc(vd a, T f, double init = 0.0)
{
    double b = init;
    rep(i, 0, a.size()) b = f(b, a[i]);
    return b;
}