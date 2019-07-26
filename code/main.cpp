#include <bits/stdc++.h>
using namespace std;

using referential=vector<double>;
using attribute=vector<referential>;
using distribution=map<int,double>;
using antecedent_weight=vector<double>;

attribute A;referential C;antecedent_weight delta;

void transform(referential& r,double& a,distribution& c)
{
	int y=lower_bound(r.begin(),r.end(),a)-r.begin(),x=y-1;
	if((y==0&&a<r[y])||y==r.size()) {}
	else if(a==r[y]) c[y]=1.0;
	else c[x]=(r[y]-a)/(r[y]-r[x]);c[y]=1.0-c[x];
}

struct rule
{
	rule(){};
	rule(vector<double> a,double c)
	{
		antecedent=vector<distribution>(A.size());
		for(int i=0;i<A.size();i++)
			transform(A[i],a[i],antecedent[i]);
		transform(C,c,consequent);
	}
	vector<distribution> antecedent;
	distribution consequent;
	double theta,w,D;
};

double individual_match_degree(distribution& a,distribution& b)
{
	return accumulate(a.begin(),a.end(),0.0,
		[&](double c,pair<int,double> _){return c+_.second*b[_.first];});
}

void calculate(rule& x,vector<rule>& r,antecedent_weight& delta)
{
	antecedent_weight norm_delta(delta);
	double max_delta=*max_element(delta.begin(),delta.end());
	for_each(norm_delta.begin(),norm_delta.end(),[&](double& d){d/=max_delta;});
	double wk,sum_wk;int k,j;
	for(k=0,sum_wk=0.0;k<r.size();r[k].w=r[k].theta*wk,sum_wk+=r[k].w,k++)
		for(j=0,wk=1.0;j<A.size();j++)
			 wk*=pow(individual_match_degree(x.antecedent[j],r[k].antecedent[j]),norm_delta[j]);
	for_each(r.begin(),r.end(),[&](rule& y){y.w/=sum_wk;});
}

vector<double> evidential_reasoning(vector<rule>& r)
{
	vector<double> beta(C.size());double N_m_D,N_m_D_,S_N_m;
	for_each(r.begin(),r.end(),[](rule& x)
	{
		x.D=1.0-x.w*accumulate(x.consequent.begin(),x.consequent.end(),0.0,
			[](double c,pair<int,double> _){return c+_.second;});
	});
	N_m_D=accumulate(r.begin(),r.end(),1.0,[](double c,rule& x){return c*x.D;});
	N_m_D_=accumulate(r.begin(),r.end(),1.0,[](double c,rule& x){return c*(1.0-x.w);});
	for(int j=0;j<C.size();j++)
		beta[j]=(accumulate(r.begin(),r.end(),1.0,
			[&](double c,rule& x){return c*(x.consequent[j]+x.D);}));
	S_N_m=accumulate(beta.begin(),beta.end(),0.0);
	for(int j=0;j<C.size();j++)
		beta[j]=(beta[j]-N_m_D)/(S_N_m-((int)C.size()-1)*N_m_D-N_m_D_);
	return beta;
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
