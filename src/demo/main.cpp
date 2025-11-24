#include "df.h"

#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    df::DataFrame df;
    df.add("c1", df::Series<int>({1, 2, 3, 4, 5}));
    std::cout << "df len: " << df.length() << "\n";
    std::cout << df.column<int>("c1") << "\n";
}