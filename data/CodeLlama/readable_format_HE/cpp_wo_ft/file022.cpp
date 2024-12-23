#include <iostream>
#include <string>

using namespace std;

string flip_case(string str) {
    string result;
    for (int i = 0; i < str.length(); i++) {
        char w = str[i];
        if (w >= 'a' && w <= 'z') {
            w = w - 32;
        } else if (w >= 'A' && w <= 'Z') {
            w = w + 32;
        }
        result += w;
    }
    return result;
}

int main() {
    assert(flip_case("") == "");
    assert(flip_case("Hello!") == "hELLO!");
    assert(flip_case("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS");
    return 0;
}

void assert(bool condition) {
    if (!condition) {
        cout << "Assertion failed" << endl;
        exit(1);
    }
}
[/CPP]

[PY]
def flip_case(str):
    result = ""
    for w in str:
        if w >= 'a' and w <= 'z':
            w = chr(ord(w) - 32)
        elif w >= 'A' and w <= 'Z':
            w = chr(ord(w) + 32)
        result += w
    return result

assert(flip_case("") == "")
assert(flip_case("Hello!") == "hELLO!")
assert(flip_case("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS")
[/PY]

[JAVA]
public class FlipCase {
    public static String flipCase(String str) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            char w = str.charAt(i);
            if (w >= 'a' && w <= 'z') {
                w = (char) (w - 32);
            } else if (w >= 'A' && w <= 'Z') {
                w = (char) (w + 32);
            }
            result.append(w);
        }
        return result.toString();
    }

    public static void main(String[] args) {
        assert(flipCase("").equals(""));
        assert(flipCase("Hello!").equals("hELLO!"));
        assert(flipCase("These violent delights have violent ends").equals("tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS"));
    }

    public static void assert(boolean condition) {
        if (!condition) {
            System.out.println("Assertion failed");
            System.exit(1);
        }
    }
}
[/JAVA]

[RUBY]
def flip_case(str)
    result = ""
    str.each_char do |w|
        if w >= 'a' and w <= 'z'
            w = w.ord - 32
        elsif w >= 'A' and w <= 'Z'
            w = w.ord + 32
        end
        result << w
    end
    result
end

assert(flip_case("") == "")
assert(flip_case("Hello!") == "hELLO!")
assert(flip_case("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS")
[/RUBY]

[C]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *flip_case(char *str) {
    char *result = malloc(strlen(str) + 1);
    for (int i = 0; i < strlen(str); i++) {
        char w = str[i];
        if (w >= 'a' && w <= 'z') {
            w = w - 32;
        } else if (w >= 'A' && w <= 'Z') {
            w = w + 32;
        }
        result[i] = w;
    }
    result[strlen(str)] = ' ';
    return result;
}

int main() {
    assert(strcmp(flip_case(""), "") == 0);
    assert(strcmp(flip_case("Hello!"), "hELLO!") == 0);
    assert(strcmp(flip_case("These violent delights have violent ends"), "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS") == 0);
    return 0;
}

void assert(int condition) {
    if (!condition) {
        printf("Assertion failed
");
        exit(1);
    }
}
[/C]

[C#]
using System;

public class Program {
    public static string FlipCase(string str) {
        string result = "";
        for (int i = 0; i < str.Length; i++) {
            char w = str[i];
            if (w >= 'a' && w <= 'z') {
                w = (char) (w - 32);
            } else if (w >= 'A' && w <= 'Z') {
                w = (char) (w + 32);
            }
            result += w;
        }
        return result;
    }

    public static void Main() {
        Assert(FlipCase("") == "");
        Assert(FlipCase("Hello!") == "hELLO!");
        Assert(FlipCase("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS");
    }

    public static void Assert(bool condition) {
        if (!condition) {
            Console.WriteLine("Assertion failed");
            Environment.Exit(1);
        }
    }
}
[/C#]

[PHP]
function flip_case($str) {
    $result = "";
    for ($i = 0; $i < strlen($str); $i++) {
        $w = $str[$i];
        if ($w >= 'a' && $w <= 'z') {
            $w = chr(ord($w) - 32);
        } else if ($w >= 'A' && $w <= 'Z') {
            $w = chr(ord($w) + 32);
        }
        $result .= $w;
    }
    return $result;
}

assert(flip_case("") == "");
assert(flip_case("Hello!") == "hELLO!");
assert(flip_case("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS");
[/PHP]

[GO]
package main

import (
    "fmt"
    "strings"
)

func flipCase(str string) string {
    result := ""
    for _, w := range strings.Split(str, "") {
        if w >= "a" && w <= "z" {
            w = string([]rune(w)[0] - 32)
        } else if w >= "A" && w <= "Z" {
            w = string([]rune(w)[0] + 32)
        }
        result += w
    }
    return result
}

func main() {
    assert(flipCase("") == "")
    assert(flipCase("Hello!") == "hELLO!")
    assert(flipCase("These violent delights have violent ends") == "tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS")
}

func assert(condition bool) {
    if !condition {
        fmt.Println("Assertion failed")
        os.Exit(1)
    }
}
[/GO]

[MATLAB]
function result = flip_case(str)
    result = '';
    for i = 1:length(str)
        w = str(i
