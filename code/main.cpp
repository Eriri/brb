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

Distribution trans_form(Referential &r, double &a)
{
	int y = lower_bound(r.begin(), r.end(), a) - r.begin(), x = y - 1;
	Distribution d;
	if ((y == 0 && a < r[y]) || y == r.size())
	{
	}
	else if (a == r[y])
		d[y] = 1.0;
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
	double tmp = accumulate(a.begin(), a.end(), 0.0, [&](double c, auto _) { return c + sqrt(_.second * b[_.first]); });
	return tmp;
	/* 
	set<int> s;
	double d = 0.0;
	for (auto &e : a)
		s.insert(e.first);
	for (auto &e : b)
		s.insert(e.first);
	for (auto &e : s)
		d += (sqrt(a[e]) - sqrt(b[e])) * (sqrt(a[e]) - sqrt(b[e]));
	return 1.0 - sqrt(d) / sqrt(2.0);
	*/
}

void normalize_delta()
{
	double D = *max_element(DELTA.begin(), DELTA.end());
	for_each(DELTA.begin(), DELTA.end(), [&](double &d) { d /= D; });
}

void activate_rule(Rule &x, vector<Rule> &r, int opt = 0, int opd = 0)
{
	for_each(r.begin(), r.end(), [&](Rule &y) {
		y.w = opt ? y.t : 1.0;
		inner_product(x.a.begin(), x.a.end(), y.a.begin(), DELTA.begin(),
					  [&](auto d, double _) { y.w*=opd?pow(_,*d):_;return ++d; },
					  [](auto &xa, auto &ya) { return match_individual(xa, ya); }); });
	double W = accumulate(r.begin(), r.end(), 0.0, [](double c, auto &y) { return c + y.w; });
	for_each(r.begin(), r.end(), [&](Rule &y) { y.w /= W; });
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
	ATTR = {{-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0},
			{-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04}};
	CONS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
	DELTA = {1.0, 1.0};
	ifstream I("../data/oil_testdata_2007.txt");
	ofstream O("result");
	double fd, pd, ls;
	double e, mse(0.0);
	vector<Rule> r, train, test, tmp;
	BeliefDistribution beta;
	while (I >> fd >> pd >> ls)
		r.push_back({{fd, pd}, ls});
	random_shuffle(r.begin(), r.end());
	train.assign(r.begin(), r.begin() + 1500);
	test.assign(r.begin() + 1500, r.end());
	for_each(test.begin(), test.end(), [&](Rule &x) {
		activate_rule(x, train);
		//tmp.clear(), tmp.reserve(train.size());
		//copy_if(train.begin(), train.end(), back_inserter(tmp), [](Rule &y) { return y.w > EPS; });
		//de(tmp.size());
		//de(tmp[0].w);
		beta = evidential_reasoning(train);
		//for (int i = 0; i < CONS.size(); i++)
		//	cout << "[" << CONS[i] << "]" << beta[i] << endl;
		e = output_result(beta);
		mse += (x.raw_c - e) * (x.raw_c - e);
		O << x.raw_a[0] << " " << x.raw_a[1] << " " << x.raw_c << " " << e << endl;
	});
	cout << mse / test.size() << endl;
}
