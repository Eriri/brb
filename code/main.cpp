#include <bits/stdc++.h>
using namespace std;
#define rep(i,s,t) for(int i=(s);i<(t);i++)
#define all(x) x.begin(),x.end()
#define sz(x) ((int)x.size())
#define fi first
#define se second

using referential=vector<double>;
using attribute=vector<referential>;
using distribution=map<int,double>;
using antecedent_weight=vector<double>;
using rule_weight=vector<double>;
using belief_degress=vector<double>;

attribute A;referential C;
antecedent_weight delta;
rule_weight w;

void transform(referential& r,double& a,distribution& c)
{
	int y=lower_bound(all(r),a)-r.begin(),x=y-1;
	if((y==0&&a<r[y])||y==sz(r)){}else if(a==r[y])c[y]=1.0;
	else c[x]=(r[y]-a)/(r[y]-r[x]);c[y]=1.0-c[x];
}

struct rule
{
	rule(){};
	rule(vector<double> a,double c)
	{
		antecedent.resize(A.size());
		rep(i,0,sz(A))transform(A[i],a[i],antecedent[i]);
		transform(C,c,consequent);
	}
	vector<distribution> antecedent;
	distribution consequent;
};

double individual_match_degree(distribution& a,distribution& b)
{
	return accumulate(all(a),0.0,[&](double c,auto _){return c+sqrt(_.se*b[_.fi]);});
}

void calculate(rule& x,vector<rule>& r,antecedent_weight& delta,rule_weight& w)
{
	antecedent_weight norm_delta(delta);double max_delta=*max_element(all(delta));
	for_each(all(norm_delta),[&](double& d){d/=max_delta;});w.assign(r.size(),1.0);
	rep(i,0,sz(r))rep(j,0,sz(A))w[i]*=pow(
		individual_match_degree(x.antecedent[j],r[i].antecedent[j]),norm_delta[j]);
	double sum_w=accumulate(all(w),0.0);
	rep(i,0,sz(w))w[i]/=sum_w;
	for_each(all(w),[&](double w){w/=sum_w;});
}

belief_degress evidential_reasoning(vector<rule>& r,rule_weight& w)
{
	double N_m_D(1.0),N_m_D_(1.0),S_N_m;
	belief_degress B(C.size(),1.0),D(r.size(),1.0);
	rep(i,0,sz(r))rep(j,0,sz(C))D[i]-=w[i]*r[i].consequent[j];
	rep(i,0,sz(r))N_m_D*=D[i],N_m_D_*=1.0-w[i];
	rep(j,0,sz(C))rep(i,0,sz(r))B[j]*=r[i].consequent[j]+D[i];
	S_N_m=accumulate(all(B),0.0);
	rep(j,0,sz(C))B[j]=(B[j]-N_m_D)/(S_N_m-(sz(C)-1)*N_m_D-N_m_D_);
	return B;
}

double similarity(distribution& a,distribution& b)
{
	double d=0.0;
	for(auto& e:a)
	return d;
}

double sra(rule& a,rule& b)
{
	double s=numeric_limits<double>::max();
	for(int i=0;i<a.antecedent.size();i++)
		s=min(s,similarity(a.antecedent[i],b.antecedent[i]));
	return s;
}

double src(rule& a,rule& b)
{
	return similarity(a.consequent,b.consequent);
}

double consistency(rule& a,rule& b)
{
	double sa=sra(a,b),sc=src(a,b);
	return exp(-sa*(1.0-sc));
}

int main()
{
	ios::sync_with_stdio(0),cout.setf(ios::fixed),cout.precision(3);

	ifstream I("../data/oil_testdata_2007.txt");
	attribute A;vector<double> B;
	A.push_back({-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0});		//flowdiff
	A.push_back({-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04});	//pressurefiff
	B={0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};					//leaksize
}
