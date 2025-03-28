#include <cstdlib>
#include <string>
#include <sstream>

int main()
{
    std::string result;

    result = circular_shift(100, 2);
    if (result != "001") {
        std::cout << "Assertion failed: circular_shift(100, 2) == '001'" << std::endl;
        return 1;
    }

    result = circular_shift(12, 2);
    if (result != "12") {
        std::cout << "Assertion failed: circular_shift(12, 2) == '12'" << std::endl;
        return 1;
    }

    result = circular_shift(97, 8);
    if (result != "79") {
        std::cout << "Assertion failed: circular_shift(97, 8) == '79'" << std::endl;
        return 1;
    }

    result = circular_shift(12, 1);
    if (result != "21") {
        std::cout << "Assertion failed: circular_shift(12, 1) == '21'" << std::endl;
        return 1;
    }

    result = circular_shift(11, 101);
    if (result != "11") {
        std::cout << "Assertion failed: circular_shift(11, 101) == '11'" << std::endl;
        return 1;
    }

    std::cout << "All tests passed." << std::endl;

    return 0;
}

std::string circular_shift(int x, int shift)
{
    std::stringstream ss;
    std::string xs, shifted;
    int len_xs, actual_shift;

    ss << x;
    ss >> xs;
    len_xs = xs.length();

    if (len_xs < shift) {
        shifted = reverse_string(xs);
    } else {
        actual_shift = shift % len_xs;
        shifted = xs.substr(len_xs-actual_shift, actual_shift) + xs.substr(0, len_xs-actual_shift);
    }

    return shifted;
}

std::string reverse_string(std::string s)
{
    std::string reversed;
    int i, len_s;

    len_s = s.length();
    for (i = 0; i < len_s; i++) {
        reversed += s[len_s-i-1];
    }

    return reversed;
}

[C]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    char result[100];

    strcpy(result, circular_shift(100, 2));
    if (strcmp(result, "001") != 0) {
        printf("Assertion failed: circular_shift(100, 2) == '001'
");
        return 1;
    }

    strcpy(result, circular_shift(12, 2));
    if (strcmp(result, "12") != 0) {
        printf("Assertion failed: circular_shift(12, 2) == '12'
");
        return 1;
    }

    strcpy(result, circular_shift(97, 8));
    if (strcmp(result, "79") != 0) {
        printf("Assertion failed: circular_shift(97, 8) == '79'
");
        return 1;
    }

    strcpy(result, circular_shift(12, 1));
    if (strcmp(result, "21") != 0) {
        printf("Assertion failed: circular_shift(12, 1) == '21'
");
        return 1;
    }

    strcpy(result, circular_shift(11, 101));
    if (strcmp(result, "11") != 0) {
        printf("Assertion failed: circular_shift(11, 101) == '11'
");
        return 1;
    }

    printf("All tests passed.
");

    return 0;
}

char *circular_shift(int x, int shift)
{
    char xs[100];
    char shifted[100];
    int len_xs, actual_shift;

    sprintf(xs, "%d", x);
    len_xs = strlen(xs);

    if (len_xs < shift) {
        strcpy(shifted, reverse_string(xs));
    } else {
        actual_shift = shift % len_xs;
        strncpy(shifted, xs+len_xs-actual_shift, actual_shift);
        strncpy(shifted+actual_shift, xs, len_xs-actual_shift);
        shifted[len_xs] = ' ';
    }

    return shifted;
}

char *reverse_string(char *s)
{
    char *reversed = malloc(strlen(s)+1);
    int i, len_s;

    len_s = strlen(s);
    for (i = 0; i < len_s; i++) {
        reversed[i] = s[len_s-i-1];
    }
    reversed[len_s] = ' ';

    return reversed;
}

[Python]
def circular_shift(x, shift):
    xs = str(x)
    len_xs = len(xs)

    if len_xs < shift:
        shifted = reverse_string(xs)
    else:
        actual_shift = shift % len_xs
        shifted = xs[len_xs-actual_shift:] + xs[:len_xs-actual_shift]

    return shifted

def reverse_string(s):
    reversed = s[::-1]
    return reversed

def main():
    result = circular_shift(100, 2)
    if result != "001":
        print("Assertion failed: circular_shift(100, 2) == '001'")
        exit(1)

    result = circular_shift(12, 2)
    if result != "12":
        print("Assertion failed: circular_shift(12, 2) == '12'")
        exit(1)

    result = circular_shift(97, 8)
    if result != "79":
        print("Assertion failed: circular_shift(97, 8) == '79'")
        exit(1)

    result = circular_shift(12, 1)
    if result != "21":
        print("Assertion failed: circular_shift(12, 1) == '21'")
        exit(1)

    result = circular_shift(11, 101)
    if result != "11":
        print("Assertion failed: circular_shift(11, 101) == '11'")
        exit(1)

    print("All tests passed.")

if __name__ == "__main__":
    main()

[Ruby]
def circular_shift(x, shift)
    xs = x.to_s
    len_xs = xs.length

    if len_xs < shift
        shifted = reverse_string(xs)
    else
        actual_shift = shift % len_xs
        shifted = xs[len_xs-actual_shift..len_xs-1] + xs[0..len_xs-actual_shift-1]
    end

    return shifted
end

def reverse_string(s)
    reversed = s.reverse
    return reversed
end

def main
    result = circular_shift(100, 2)
    if result != "001"
        puts "Assertion failed: circular_shift(100, 2) == '001'"
        exit(1)
    end

    result = circular_shift(12, 2)
    if result != "12"
        puts "Assertion failed: circular_shift(12,
