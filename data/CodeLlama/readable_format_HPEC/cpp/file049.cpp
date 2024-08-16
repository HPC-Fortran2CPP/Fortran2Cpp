#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    const int dp = 8; // Assuming double precision
    std::vector<double> base(2025, 0.0);
    std::vector<double*> xa1(2025, nullptr);
    std::vector<double*> xa2(2025, nullptr);

    // Initialize xa1 and xa2 to point to the base array
    for (int i = 0; i < 2025; ++i) {
        xa1[i] = &base[i];
        xa2[i] = &base[i];
    }

    int n = 180;
    std::vector<int> indexSet = {
        521, 523, 525, 527, 529, 531, 547, 549, 
        551, 553, 555, 557, 573, 575, 577, 579, 581, 583, 599, 
        601, 603, 605, 607, 609, 625, 627, 629, 631, 633, 635, 
        651, 653, 655, 657, 659, 661, 859, 861, 863, 865, 867, 
        869, 885, 887, 889, 891, 893, 895, 911, 913, 915, 917, 
        919, 921, 937, 939, 941, 943, 945, 947, 963, 965, 967, 
        969, 971, 973, 989, 991, 993, 995, 997, 999, 1197, 1199, 
        1201, 1203, 1205, 1207, 1223, 1225, 1227, 1229, 1231, 
        1233, 1249, 1251, 1253, 1255, 1257, 1259, 1275, 1277, 
        1279, 1281, 1283, 1285, 1301, 1303, 1305, 1307, 1309, 
        1311, 1327, 1329, 1331, 1333, 1335, 1337, 1535, 1537, 
        1539, 1541, 1543, 1545, 1561, 1563, 1565, 1567, 1569, 
        1571, 1587, 1589, 1591, 1593, 1595, 1597, 1613, 1615, 
        1617, 1619, 1621, 1623, 1639, 1641, 1643, 1645, 1647, 
        1649, 1665, 1667, 1669, 1671, 1673, 1675, 1873, 1875, 
        1877, 1879, 1881, 1883, 1899, 1901, 1903, 1905, 1907, 
        1909, 1925, 1927, 1929, 1931, 1933, 1935, 1951, 1953, 
        1955, 1957, 1959, 1961, 1977, 1979, 1981, 1983, 1985, 
        1987, 2003, 2005, 2007, 2009, 2011, 2013
    };

    // Populate base array
    for (int i = 0; i < 2025; ++i) {
        base[i] = 0.5 * (i + 1); // Adjusted for 0-based indexing
    }

    // Parallel region
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int idx1 = indexSet[i] - 1; // Adjust for 0-based indexing
        int idx2 = indexSet[i] + 11 - 1; // Adjust for 0-based indexing
        base[idx1] += 1.0;
        base[idx2] += 3.0;
    }

    // Print results
    std::cout << "xa1[999] = " << base[999 - 1] << " xa2[1285] = " << base[1285 - 1] << std::endl;

    // Clean up
    xa1.clear();
    xa2.clear();

    return 0;
}
