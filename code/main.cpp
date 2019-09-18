#include <bits/stdc++.h>
#include "tpc.cpp"
using namespace std;
#define rep(i, s, t) for (auto i = (s); i < (t); i++)
#define repb(i, s, t) for (auto i = (s); i <= (t); i++)
#define all(x) x.begin(), x.end()
#define dd(x) cout << #x << '=' << x << ' '
#define de(x) cout << #x << '=' << x << '\n'

struct rule
{
	vd a, c;
};

vd o, l;

vd er(vector<rule> rb, vd v)
{
	vd aw = vd(rb.size());
	rep(i, 0, rb.size())
		aw[i] = afunc(ufunc<uf>(0.5 * ((rb[i].a - v) * (rb[i].a - v) / (o * o)), exp), multiplies(), 1.0);
	double sw = afunc(aw, plus());
	if (sw == 0.0)
		return urd(l.size());
	if (sw == *max_element(all(aw)))
		return rb[max_element(all(aw)) - aw.begin()].c;
	rep(j, 0, l.size())
	{
		}
}

int main()
{
	ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
	cout.setf(ios::fixed), cout.precision(3);
	vd a = {1, 2, 3, 4};
	a = ufunc<uf>(a, exp);
	rep(i, 0, a.size()) cout << a[i] << endl;
}
