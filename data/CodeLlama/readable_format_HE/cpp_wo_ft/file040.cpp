#include <cstdlib>
#include <string>

bool is_palindrome(const std::string& text)
{
    std::string reversed_text = text;
    std::reverse(reversed_text.begin(), reversed_text.end());
    return reversed_text == text;
}

int main()
{
    if (!is_palindrome("")) {
        std::cerr << "Assertion failed: is_palindrome('') == true" << std::endl;
        return EXIT_FAILURE;
    }

    if (!is_palindrome("aba")) {
        std::cerr << "Assertion failed: is_palindrome('aba') == true" << std::endl;
        return EXIT_FAILURE;
    }

    if (!is_palindrome("aaaaa")) {
        std::cerr << "Assertion failed: is_palindrome('aaaaa') == true" << std::endl;
        return EXIT_FAILURE;
    }

    if (is_palindrome("zbcd")) {
        std::cerr << "Assertion failed: is_palindrome('zbcd') == false" << std::endl;
        return EXIT_FAILURE;
    }

    if (!is_palindrome("xywyx")) {
        std::cerr << "Assertion failed: is_palindrome('xywyx') == true" << std::endl;
        return EXIT_FAILURE;
    }

    if (is_palindrome("xywyz")) {
        std::cerr << "Assertion failed: is_palindrome('xywyz') == false" << std::endl;
        return EXIT_FAILURE;
    }

    if (is_palindrome("xywzx")) {
        std::cerr << "Assertion failed: is_palindrome('xywzx') == false" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed." << std::endl;
    return EXIT_SUCCESS;
}

[C]
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

bool is_palindrome(const char* text)
{
    size_t n = strlen(text);
    char* reversed_text = malloc(n + 1);
    if (reversed_text == NULL) {
        return false;
    }

    for (size_t i = 0; i < n; i++) {
        reversed_text[i] = text[n - i - 1];
    }
    reversed_text[n] = ' ';

    bool result = strcmp(reversed_text, text) == 0;
    free(reversed_text);
    return result;
}

int main()
{
    if (!is_palindrome("")) {
        fprintf(stderr, "Assertion failed: is_palindrome('') == true
");
        return EXIT_FAILURE;
    }

    if (!is_palindrome("aba")) {
        fprintf(stderr, "Assertion failed: is_palindrome('aba') == true
");
        return EXIT_FAILURE;
    }

    if (!is_palindrome("aaaaa")) {
        fprintf(stderr, "Assertion failed: is_palindrome('aaaaa') == true
");
        return EXIT_FAILURE;
    }

    if (is_palindrome("zbcd")) {
        fprintf(stderr, "Assertion failed: is_palindrome('zbcd') == false
");
        return EXIT_FAILURE;
    }

    if (!is_palindrome("xywyx")) {
        fprintf(stderr, "Assertion failed: is_palindrome('xywyx') == true
");
        return EXIT_FAILURE;
    }

    if (is_palindrome("xywyz")) {
        fprintf(stderr, "Assertion failed: is_palindrome('xywyz') == false
");
        return EXIT_FAILURE;
    }

    if (is_palindrome("xywzx")) {
        fprintf(stderr, "Assertion failed: is_palindrome('xywzx') == false
");
        return EXIT_FAILURE;
    }

    printf("All tests passed.
");
    return EXIT_SUCCESS;
}

[Python]
def is_palindrome(text):
    return text == text[::-1]


def main():
    assert is_palindrome("")
    assert is_palindrome("aba")
    assert is_palindrome("aaaaa")
    assert not is_palindrome("zbcd")
    assert is_palindrome("xywyx")
    assert not is_palindrome("xywyz")
    assert not is_palindrome("xywzx")
    print("All tests passed.")


if __name__ == "__main__":
    main()

[Java]
public class Palindrome {
    public static boolean isPalindrome(String text) {
        return text.equals(new StringBuilder(text).reverse().toString());
    }

    public static void main(String[] args) {
        assert isPalindrome("");
        assert isPalindrome("aba");
        assert isPalindrome("aaaaa");
        assert !isPalindrome("zbcd");
        assert isPalindrome("xywyx");
        assert !isPalindrome("xywyz");
        assert !isPalindrome("xywzx");
        System.out.println("All tests passed.");
    }
}

[JavaScript]
function isPalindrome(text) {
    return text === text.split("").reverse().join("");
}

function main() {
    assert(isPalindrome(""));
    assert(isPalindrome("aba"));
    assert(isPalindrome("aaaaa"));
    assert(!isPalindrome("zbcd"));
    assert(isPalindrome("xywyx"));
    assert(!isPalindrome("xywyz"));
    assert(!isPalindrome("xywzx"));
    console.log("All tests passed.");
}

main();

[PHP]
function isPalindrome($text)
{
    return $text === strrev($text);
}

function main()
{
    assert(isPalindrome(""));
    assert(isPalindrome("aba"));
    assert(isPalindrome("aaaaa"));
    assert(!isPalindrome("zbcd"));
    assert(isPalindrome("xywyx"));
    assert(!isPalindrome("xywyz"));
    assert(!isPalindrome("xywzx"));
    echo "All tests passed.
";
}

main();

[Ruby]
def is_palindrome(text)
    text == text.reverse
end

def main
    assert(is_palindrome(""))
    assert(is_palindrome("aba"))
    assert(is_palindrome("aaaaa"))
    assert(!is_palindrome("zbcd"))
    assert(is_palindrome("xywyx"))
    assert(!is_palindrome("xywyz"))
    assert(!is_palindrome("xywzx"))
    puts "All tests passed."
end

main

[Swift]
func isPalindrome(_ text: String) -> Bool {
    return text == String(text.reversed())
}

func main() {
    assert(isPalindrome(""))
    assert(isPalindrome("aba"))
    assert(isPalindrome("aaaaa"))
    assert(!isPalindrome("zbcd"))
    assert(isPalindrome("xywyx"))
    assert(!isPalindrome("xywyz"))
    assert(!isPalindrome("xywzx"))
    print("All tests passed.")
}

main()

[Go]
func isPalindrome(text string) bool {
    return text == string([]rune(text)[len(text)-1:0:-1])
}

func main() {
    assert(isPalindrome(""))
    assert(isPalind
