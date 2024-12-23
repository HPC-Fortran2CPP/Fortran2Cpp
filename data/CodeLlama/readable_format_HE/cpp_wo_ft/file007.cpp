#include <iostream>
#include <string>
#include <cassert>

using namespace std;

bool is_palindrome(const string& str) {
    string reversed_str = str;
    reverse(reversed_str.begin(), reversed_str.end());
    return (str == reversed_str);
}

string make_palindrome(const string& str) {
    string palindrome_str;
    string rstr, nstr, n2str;
    int i;

    for (i = 1; i <= str.length(); i++) {
        rstr = str.substr(i, str.length() - i + 1);
        if (is_palindrome(rstr)) {
            nstr = str.substr(0, i - 1);
            n2str = reverse_string(nstr);
            palindrome_str = str + n2str;
            return palindrome_str;
        }
    }

    n2str = reverse_string(str);
    palindrome_str = str + n2str;
    return palindrome_str;
}

string reverse_string(const string& str) {
    string reversed_str;
    int i;

    for (i = 1; i <= str.length(); i++) {
        reversed_str += str.substr(str.length() - i + 1, 1);
    }

    return reversed_str;
}

int main() {
    assert(make_palindrome("") == "");
    assert(make_palindrome("x") == "x");
    assert(make_palindrome("xyz") == "xyzyx");
    assert(make_palindrome("xyx") == "xyx");
    assert(make_palindrome("jerry") == "jerryrrej");

    return 0;
}

[C]
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

bool is_palindrome(const char* str) {
    char* reversed_str = strdup(str);
    reverse_string(reversed_str);
    bool result = (strcmp(str, reversed_str) == 0);
    free(reversed_str);
    return result;
}

char* make_palindrome(const char* str) {
    char* palindrome_str = malloc(strlen(str) * 2 + 1);
    char* rstr;
    char* nstr;
    char* n2str;
    int i;

    for (i = 1; i <= strlen(str); i++) {
        rstr = malloc(strlen(str) - i + 1);
        strncpy(rstr, str + i, strlen(str) - i);
        rstr[strlen(str) - i] = ' ';
        if (is_palindrome(rstr)) {
            nstr = malloc(i);
            strncpy(nstr, str, i - 1);
            nstr[i - 1] = ' ';
            n2str = reverse_string(nstr);
            strcpy(palindrome_str, str);
            strcat(palindrome_str, n2str);
            free(rstr);
            free(nstr);
            free(n2str);
            return palindrome_str;
        }
        free(rstr);
    }

    n2str = reverse_string(str);
    strcpy(palindrome_str, str);
    strcat(palindrome_str, n2str);
    free(n2str);
    return palindrome_str;
}

char* reverse_string(const char* str) {
    char* reversed_str = malloc(strlen(str) + 1);
    int i;

    for (i = 1; i <= strlen(str); i++) {
        reversed_str[i - 1] = str[strlen(str) - i];
    }
    reversed_str[i - 1] = ' ';

    return reversed_str;
}

int main() {
    assert(strcmp(make_palindrome(""), "") == 0);
    assert(strcmp(make_palindrome("x"), "x") == 0);
    assert(strcmp(make_palindrome("xyz"), "xyzyx") == 0);
    assert(strcmp(make_palindrome("xyx"), "xyx") == 0);
    assert(strcmp(make_palindrome("jerry"), "jerryrrej") == 0);

    return 0;
}

[Python]
def is_palindrome(str):
    return str == str[::-1]

def make_palindrome(str):
    palindrome_str = ""
    rstr = ""
    nstr = ""
    n2str = ""
    for i in range(1, len(str)):
        rstr = str[i:]
        if is_palindrome(rstr):
            nstr = str[:i]
            n2str = reverse_string(nstr)
            palindrome_str = str + n2str
            return palindrome_str
    n2str = reverse_string(str)
    palindrome_str = str + n2str
    return palindrome_str

def reverse_string(str):
    reversed_str = ""
    for i in range(1, len(str)):
        reversed_str += str[len(str) - i]
    return reversed_str

assert(make_palindrome("") == "")
assert(make_palindrome("x") == "x")
assert(make_palindrome("xyz") == "xyzyx")
assert(make_palindrome("xyx") == "xyx")
assert(make_palindrome("jerry") == "jerryrrej")

[Ruby]
def is_palindrome(str)
    str == str.reverse
end

def make_palindrome(str)
    palindrome_str = ""
    rstr = ""
    nstr = ""
    n2str = ""
    for i in 1..str.length
        rstr = str[i..-1]
        if is_palindrome(rstr)
            nstr = str[0..i-1]
            n2str = reverse_string(nstr)
            palindrome_str = str + n2str
            return palindrome_str
        end
    end
    n2str = reverse_string(str)
    palindrome_str = str + n2str
    return palindrome_str
end

def reverse_string(str)
    reversed_str = ""
    for i in 1..str.length
        reversed_str += str[str.length - i]
    end
    return reversed_str
end

assert(make_palindrome("") == "")
assert(make_palindrome("x") == "x")
assert(make_palindrome("xyz") == "xyzyx")
assert(make_palindrome("xyx") == "xyx")
assert(make_palindrome("jerry") == "jerryrrej")

[JavaScript]
function isPalindrome(str) {
    return str === str.split('').reverse().join('');
}

function makePalindrome(str) {
    let palindromeStr = "";
    let rstr = "";
    let nstr = "";
    let n2str = "";
    for (let i = 1; i <= str.length; i++) {
        rstr = str.substring(i);
        if (isPalindrome(rstr)) {
            nstr = str.substring(0, i - 1);
            n2str = reverseString(nstr);
            palindromeStr = str + n2str;
            return palindromeStr;
        }
    }
    n2str = reverseString(str);
    palindromeStr = str + n2str;
    return palindromeStr;
}

function reverseString(str) {
    let reversedStr = "";
    for (let i = 1; i <= str.length; i++) {
        reversedStr +=
