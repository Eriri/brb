#include <bits/stdc++.h>
using namespace std;

#define dd(x) cout << #x << '=' << x << ' '
#define de(x) cout << #x << '=' << x << '\n'

struct e
{
    double fd, pd, ls;
};

int main()
{
    ifstream I("../data/titanic/train_rev.txt");
    double survived, pclass, age, sib, par;
    string sex, emb;
    double minage = numeric_limits<double>::max();
    double maxage = numeric_limits<double>::min();
    double minsib = numeric_limits<double>::max();
    double maxsib = numeric_limits<double>::min();
    double minpar = numeric_limits<double>::max();
    double maxpar = numeric_limits<double>::min();
    while (I >> survived >> pclass >> sex >> age >> sib >> par >> emb)
    {
        minage = min(minage, age);
        maxage = max(maxage, age);
        minsib = min(minsib, sib);
        maxsib = max(maxsib, sib);
        minpar = min(minpar, par);
        maxpar = max(maxpar, par);
    }
    dd(minage), de(maxage);
    dd(minsib), de(maxsib);
    dd(minpar), de(maxpar);
}