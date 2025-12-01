#include <iostream>
#include "df.h"

int main() {
    df::Series<int> s1({1, 2, 3, 4, 5});
    df::Series<int> s2({5, 4, 3, 2, 1});

    auto s3 = s1 + s2;
    s3.set_null(2);
    std::cout << "s3: " << s3 << std::endl;

    return 0;
}