#include <bits/stdc++.h>
using namespace std;
#define dd(x) cout << #x << '=' << x << ' '
#define de(x) cout << #x << '=' << x << '\n'

using Referential = vector<double>;
using Distribution = map<int, double>;

using Attribute = vector<Referential>;
using Antecedent = vector<Distribution>;
using Consequent = Distribution;

using AntecedentWeight = vector<double>;
using RuleWeight = double;
using ActivateWeight = double;
using BeliefDegress = double;
using BeliefDistribution = vector<BeliefDegress>;

Attribute ATTR;
Referential CONS;
AntecedentWeight DELTA;
const double EPS = 1e-5;
const double TEPS = 1e-4;
const double AEPS = 1e-3;

Distribution trans_form(Referential &r, double &a)
{
	int y = lower_bound(r.begin(), r.end(), a) - r.begin(), x = y - 1;
	Distribution d;
	if ((y == 0 && r[y] - a > TEPS) || (y == r.size() && a - r[x] > TEPS))
	{
	}
	else if (y != r.size() && r[y] - a <= TEPS)
		d[y] = 1.0;
	else if (x != -1 && a - r[x] <= TEPS)
		d[x] = 1.0;
	else
		d[x] = (r[y] - a) / (r[y] - r[x]),
		d[y] = 1.0 - d[x];
	return d;
}

struct Rule
{
	Rule(){};
	Rule(vector<double> ar, double cs)
	{
		a.resize(ATTR.size());
		inner_product(ATTR.begin(), ATTR.end(), ar.begin(), a.begin(),
					  [](auto ai, auto d) {*ai=d;return ++ai; }, trans_form);
		c = trans_form(CONS, cs);
		raw_a = ar, raw_c = cs;
		t = 1.0, w = 1.0;
	}
	Antecedent a;
	Consequent c;
	RuleWeight t;
	ActivateWeight w;
	BeliefDegress d;
	vector<double> raw_a;
	double raw_c;
};

double match_individual(Distribution &a, Distribution b)
{
	return 1.0 - sqrt(1.0 - accumulate(a.begin(), a.end(), 0.0,
									   [&](double c, auto _) { return c + sqrt(_.second * b[_.first]); }));
	/* 
	set<int> s;
	double d = 0.0;
	for (auto &e : a)
		s.insert(e.first);
	for (auto &e : b)
		s.insert(e.first);
	for (auto &e : s)
		d += (sqrt(a[e]) - sqrt(b[e])) * (sqrt(a[e]) - sqrt(b[e]));
	return 1.0 - sqrt(d) / sqrt(2.0);*/
}

void normalize_delta()
{
	double D = *max_element(DELTA.begin(), DELTA.end());
	for_each(DELTA.begin(), DELTA.end(), [&](double &d) { d /= D; });
}

void activate_rule(Rule &x, vector<Rule> &r, int opn = 1, int opt = 0, int opd = 0)
{
	for_each(r.begin(), r.end(), [&](Rule &y) {
		y.w = opt ? y.t : 1.0;
		inner_product(x.a.begin(), x.a.end(), y.a.begin(), DELTA.begin(),
					  [&](auto d, double _) { y.w*=opd?pow(_,*d):_;return ++d; },
					  [](auto &xa, auto &ya) { return match_individual(xa, ya); }); });
	double W = accumulate(r.begin(), r.end(), 0.0, [](double c, auto &y) { return c + y.w; });
	for_each(r.begin(), r.end(), [&](Rule &y) { y.w /= opn ? W : 1.0; });
}

BeliefDistribution evidential_reasoning(vector<Rule> &r)
{
	BeliefDistribution beta(CONS.size(), 1.0);
	BeliefDegress rd(1.0), ad(1.0), sd(0.0);
	for_each(r.begin(), r.end(),
			 [&](Rule &x) {
				 x.d = 1.0 - x.w * accumulate(x.c.begin(), x.c.end(), 0.0,
											  [](double c, auto &_) { return c + _.second; });
				 rd *= x.d, ad *= 1.0 - x.w;
				 for (int i = 0; i < CONS.size(); i++)
					 beta[i] *= x.w * x.c[i] + x.d;
			 });
	sd = accumulate(beta.begin(), beta.end(), 0.0);
	for_each(beta.begin(), beta.end(),
			 [&](BeliefDegress &b) { b = (b - rd) / (sd - rd * ((int)CONS.size() - 1) - ad); });
	return beta;
}

double output_result(BeliefDistribution &beta)
{
	double o = 0.0;
	accumulate(beta.begin(), beta.end(), CONS.begin(),
			   [&](auto c, BeliefDegress &b) {o+=*c*b;return ++c; });
	return o;
}

double measure_antecedent(Antecedent &a, Antecedent &b)
{
	return inner_product(a.begin(), a.end(), b.begin(), numeric_limits<double>::max(),
						 [](double c, double _) { return min(c, _); },
						 [](Distribution &da, Distribution &db) { return match_individual(da, db); });
}

double measure_consequent(Consequent &a, Consequent &b)
{
	return match_individual(a, b);
}

double calculate_consistency(Rule &a, Rule &b)
{
	return exp(measure_antecedent(a.a, b.a) * (1.0 - measure_consequent(a.c, b.c)));
}

void calculate_consistency(vector<Rule> &r)
{
}

int main()
{
	ios::sync_with_stdio(0), cout.setf(ios::fixed), cout.precision(3);

	ATTR = {{-11, -6, -3, -1, 0, 1, 2, 3},
			{-0.061, -0.005, -0.002, 0.000, 0.005, 0.010, 0.060}};
	CONS = {0, 2, 4, 6, 8};

	DELTA.resize(ATTR.size(), 1.0);
	ifstream I("../data/oil_rev.txt");
	ofstream O("result");
	BeliefDistribution beta;
	vector<Rule> r, train, test;
	double fd, pd, ls;
	double e, mae, mse;
	while (I >> fd >> pd >> ls)
		r.push_back(Rule({fd, pd}, ls));
	train.assign(r.begin(), r.begin() + 1500);
	test.assign(r.begin() + 1500, r.end());
	for_each(test.begin(), test.end(), [&](Rule &x) {
		activate_rule(x, train);
		beta = evidential_reasoning(train);
		e = output_result(beta);
		mse += (x.raw_c - e) * (x.raw_c - e);
		mae += fabs(x.raw_c - e);
		O << x.raw_a[0] << " " << x.raw_a[1] << " " << x.raw_c << " " << e << endl;
	});
	cout << "mse_sum: " << mse / test.size() << endl;
	cout << "mae_sum: " << mae / test.size() << endl;
}
