/* =============================================================================
 *  pda_expression_validator.cpp
 *
 *  AUTOMATA: Automata Theory  --  Scenario 5: Arithmetic Expression Validator
 *
 *  A single-file console application that decides whether a given string is a
 *  syntactically valid arithmetic expression, by simulating a DETERMINISTIC
 *  PUSH-DOWN AUTOMATON (DPDA).
 *
 *  THE LANGUAGE
 *  -------------
 *  Sigma = { 0,1,2,3,4,5,6,7,8,9, +, -, *, /, (, ) }          (|Sigma| = 16)
 *
 *  L = the set of all strings over Sigma that
 *      (1) have balanced and properly nested parentheses,
 *      (2) contain no operator at the start or at the end,
 *      (3) contain no two consecutive operators,
 *      (4) use a unary minus only at the start of the expression, immediately
 *          after an opening parenthesis, or immediately after another operator,
 *      (5) contain no digit sequence with a leading zero unless the sequence
 *          is exactly "0".
 *
 *  THE AUTOMATON
 *  -------------
 *  M = (Q, Sigma, Gamma, delta, q0, Z0, F)
 *
 *    Q     = { q0, q1, q2, qD }                     |Q| = 4
 *    Gamma = { Z, '(' }                             |Gamma| = 2
 *    q0    = q0     (start state, "expecting an operand")
 *    Z0    = Z     (bottom-of-stack marker)
 *    F     = { q2 } (accepting state, "expecting an operator")
 *
 *    q0 : expecting an operand -- a digit, a unary '-' or '(' may follow.
 *          (also the state reached right after a binary operator)
 *    q1 : reading a number that began with a digit 1-9; more digits may follow.
 *    q2 : a complete number or a closed group was read; only a binary
 *          operator, ')' or the end of the input may follow.   (ACCEPTING)
 *    qD : dead state -- any further input is rejected.
 *
 *  The transition function delta is stored explicitly in the table
 *  `delta[state][symbol][stack-top]` below; the third index is the symbol
 *  currently on top of the stack, which is what lets one step of the machine
 *  check the balance and the nesting of the parentheses.
 *
 *  A special end-of-input marker (SYM_END, not part of Sigma) is appended to
 *  the input by the simulator, so the machine is a strict DPDA (no epsilon
 *  moves): in q1 the marker closes the number being read, in q2 it is
 *  harmless, and in q0 it is an error (the expression ended while an operand
 *  was still expected).
 *
 *  BUILD & RUN
 *  -----------
 *    g++ -std=c++17 -Wall -Wextra -pedantic pda_expression_validator.cpp \
 *        -o pda_expression_validator.exe
 *
 *    ./pda_expression_validator.exe              (interactive menu)
 *    ./pda_expression_validator.exe "3+4"       (one expression from argv)
 *    ./pda_expression_validator.exe --test       (built-in test suite)
 * ============================================================================= */

#include <cstddef>   /* std::size_t                                   */
#include <iostream>  /* std::cout, std::cin                           */
#include <limits>    /* std::numeric_limits                           */
#include <string>    /* std::string                                   */
#include <vector>    /* std::vector (used as the PDA stack)            */

/* ===========================================================================
 *  1.  THE ALPHABET, THE STATES AND THE STACK ALPHABET
 * ======================================================================== */

/* Input symbols.  The order of this enumeration IS the order of the columns
 * of the transition table, so the table can be read straight off the screen. */
enum Symbol {
    SYM_0 = 0, SYM_1, SYM_2, SYM_3, SYM_4, SYM_5, SYM_6, SYM_7, SYM_8,
    SYM_9,          /*  0  1  2  3  4  5  6  7  8  9                       */
    SYM_PLUS,       /* 10  +                                            */
    SYM_MINUS,      /* 11  -                                            */
    SYM_STAR,       /* 12  *                                            */
    SYM_SLASH,      /* 13  /                                            */
    SYM_LPAREN,     /* 14  (                                            */
    SYM_RPAREN,     /* 15  )                                            */
    SYM_END,        /* 16  end-of-input marker (not part of Sigma)        */
    NUM_SYMBOLS
};

/* Control states of the PDA. */
enum State {
    ST_EXPECT_OPERAND = 0,  /* q0 - start state                            */
    ST_READING_NUMBER,      /* q1 - inside a number that began with 1-9   */
    ST_EXPECT_OPERATOR,     /* q2 - accepting state                        */
    ST_DEAD,                /* qD - trap state                             */
    NUM_STATES
};

/* Stack alphabet.  The stack holds one marker for every open parenthesis. */
enum StackSymbol {
    ST_BOTTOM = 0,   /* Z  - the bottom-of-stack marker, never popped     */
    ST_LPAREN,       /* '(' - one open parenthesis                        */
    NUM_STACK_SYMBOLS
};

/* What a transition does to the stack. */
enum Action {
    ACT_KEEP = 0,    /* pop the top symbol and push it back (no change)   */
    ACT_POP,         /* pop the top symbol and push nothing (epsilon)     */
    ACT_PUSH_LPAREN  /* keep the top symbol and push a '(' on top of it  */
};

/* One entry of the transition function. */
struct Transition {
    int nextState;
    int action;
};

/* ===========================================================================
 *  2.  THE TRANSITION FUNCTION
 * ======================================================================== */

/* Each table cell holds the two entries that correspond to the two possible
 * top-of-stack symbols: index 0 = Z (no open parenthesis), index 1 = '('. */
#define T(next, act)   { { (next), (act) }, { (next), (act) } }
#define T_DEAD         { { ST_DEAD, ACT_KEEP }, { ST_DEAD, ACT_KEEP } }
/* A ')' needs a matching '(' on the stack: with '(' on top we pop it and
 * carry on, with only the bottom marker Z on top we are unbalanced and die.
 * Index 0 of a cell is the case "top of stack is Z", index 1 is "top is '('". */
#define T_CLOSE(next)  { { ST_DEAD, ACT_KEEP }, { (next), ACT_POP } }

static const Transition delta[NUM_STATES][NUM_SYMBOLS][NUM_STACK_SYMBOLS] = {

    /* ---------------------------------------------------------------------
     *  q0  --  EXPECTING AN OPERAND
     *  A digit, a unary '-' or an opening parenthesis may legally follow.
     * ------------------------------------------------------------------- */
    {
        T(ST_EXPECT_OPERATOR, ACT_KEEP),    /*  '0'  a lone zero: no more
                                                digits may follow, so the
                                                leading-zero rule is
                                                enforced by the target state */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '1' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '2' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '3' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '4' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '5' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '6' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '7' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '8' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '9' */
        T_DEAD,                              /*  '+'  operator where an
                                                operand is required         */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '-'  unary minus           */
        T_DEAD,                              /*  '*' */
        T_DEAD,                              /*  '/' */
        T(ST_EXPECT_OPERAND, ACT_PUSH_LPAREN), /* '('  push one marker     */
        T_DEAD,                              /*  ')'  nothing to close      */
        T_DEAD                               /*  END  operand was missing   */
    },

    /* ---------------------------------------------------------------------
     *  q1  --  READING A NUMBER (the number began with a digit 1-9)
     * ------------------------------------------------------------------- */
    {
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '0' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '1' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '2' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '3' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '4' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '5' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '6' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '7' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '8' */
        T(ST_READING_NUMBER, ACT_KEEP),      /*  '9' */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '+'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '-'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '*'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '/'  binary operator       */
        T_DEAD,                              /*  '('  implicit multiplication
                                                is not allowed                */
        T_CLOSE(ST_EXPECT_OPERATOR),         /*  ')'  close the group       */
        T(ST_EXPECT_OPERATOR, ACT_KEEP)       /*  END  the number is complete */
    },

    /* ---------------------------------------------------------------------
     *  q2  --  EXPECTING AN OPERATOR  (the accepting state)
     * ------------------------------------------------------------------- */
    {
        T_DEAD,                              /*  '0'  a digit here is either a
                                                leading zero or an implicit
                                                multiplication               */
        T_DEAD,                              /*  '1' */
        T_DEAD,                              /*  '2' */
        T_DEAD,                              /*  '3' */
        T_DEAD,                              /*  '4' */
        T_DEAD,                              /*  '5' */
        T_DEAD,                              /*  '6' */
        T_DEAD,                              /*  '7' */
        T_DEAD,                              /*  '8' */
        T_DEAD,                              /*  '9' */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '+'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '-'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '*'  binary operator       */
        T(ST_EXPECT_OPERAND, ACT_KEEP),      /*  '/'  binary operator       */
        T_DEAD,                              /*  '('  implicit multiplication
                                                is not allowed                */
        T_CLOSE(ST_EXPECT_OPERATOR),         /*  ')'  close the group       */
        T(ST_EXPECT_OPERATOR, ACT_KEEP)       /*  END  a complete expression */
    },

    /* ---------------------------------------------------------------------
     *  qD  --  DEAD STATE: everything leads back to qD.
     * ------------------------------------------------------------------- */
    {
        T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD,
        T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD, T_DEAD,
        T_DEAD
    }
};

#undef T
#undef T_DEAD
#undef T_CLOSE

/* ===========================================================================
 *  3.  THE SIMULATOR
 * ======================================================================== */

/* Map a character to its symbol index, or -1 when the character is not in
 * Sigma.  The comparison is written out by hand (instead of using <cctype>)
 * so that bytes with the high bit set cannot trigger undefined behaviour. */
static int symbol_index(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';

    switch (c) {
    case '+': return SYM_PLUS;
    case '-': return SYM_MINUS;
    case '*': return SYM_STAR;
    case '/': return SYM_SLASH;
    case '(': return SYM_LPAREN;
    case ')': return SYM_RPAREN;
    default:  return -1;                    /* not in Sigma */
    }
}

/* Render the stack (bottom of the stack on the left) for the trace output. */
static std::string stack_to_string(const std::vector<int>& stack)
{
    std::string text;
    std::size_t i;

    for (i = 0; i < stack.size(); i++) {
        if (stack[i] == ST_BOTTOM)
            text += 'Z';
        else
            text += '(';
    }
    return text;
}

/* Name of a state, for the trace output. */
static const char* state_name(int state)
{
    switch (state) {
    case ST_EXPECT_OPERAND: return "q0";
    case ST_READING_NUMBER: return "q1";
    case ST_EXPECT_OPERATOR: return "q2";
    default:                return "qD";
    }
}

/* Name of an input symbol, for the trace output. */
static std::string symbol_name(int symbol)
{
    switch (symbol) {
    case SYM_0:      return "0";
    case SYM_1:      return "1";
    case SYM_2:      return "2";
    case SYM_3:      return "3";
    case SYM_4:      return "4";
    case SYM_5:      return "5";
    case SYM_6:      return "6";
    case SYM_7:      return "7";
    case SYM_8:      return "8";
    case SYM_9:      return "9";
    case SYM_PLUS:   return "+";
    case SYM_MINUS:  return "-";
    case SYM_STAR:   return "*";
    case SYM_SLASH:  return "/";
    case SYM_LPAREN: return "(";
    case SYM_RPAREN: return ")";
    default:         return "END";
    }
}

/* Apply the stack action of a transition to the PDA stack.
   The table only asks for a POP when an '(' is on top of the stack, so the
   bottom marker Z is never removed; the guard is kept as a safety net. */
static void apply_action(std::vector<int>& stack, int action)
{
    switch (action) {
    case ACT_POP:
        if (!stack.empty() && stack.back() != ST_BOTTOM)
            stack.pop_back();
        break;
    case ACT_PUSH_LPAREN:
        stack.push_back(ST_LPAREN);
        break;
    case ACT_KEEP:
    default:
        break;
    }
}

/* ---------------------------------------------------------------------------
 *  THE PUBLIC API
 *
 *  bool accepts(std::string s) -> true  if s belongs to the language L,
 *                                  false otherwise.
 *
 *  This is the single function the rest of the program is built on; it is
 *  declared at namespace scope so that it has external linkage and can be
 *  reused by any other translation unit that links against this file.
 * ------------------------------------------------------------------------- */
bool accepts(std::string s)
{
    std::vector<int> stack;        /* the PDA stack; stack[0] is always Z   */
    int state;
    std::size_t i;

    stack.push_back(ST_BOTTOM);
    state = ST_EXPECT_OPERAND;

    /* The loop also processes the end-of-input marker, which is the symbol
     * at position s.size(). */
    for (i = 0; i <= s.size(); i++) {
        int symbol = (i < s.size()) ? symbol_index(s[i]) : SYM_END;
        Transition step;

        if (symbol < 0)
            return false;                 /* a character outside Sigma */

        step = delta[state][symbol][stack.back()];

        apply_action(stack, step.action);
        state = step.nextState;

        if (state == ST_DEAD)
            return false;                 /* the string broke a rule */
    }

    /* Accept only a complete expression whose parentheses are all closed. */
    return state == ST_EXPECT_OPERATOR &&
           stack.size() == 1 &&
           stack.back() == ST_BOTTOM;
}

/* ---------------------------------------------------------------------------
 *  TRACE: print the sequence of configurations of the PDA.
 *  A configuration is written as (state, unread input, stack).
 * ------------------------------------------------------------------------- */
static void trace(const std::string& s)
{
    std::vector<int> stack;
    int state;
    std::size_t i;

    stack.push_back(ST_BOTTOM);
    state = ST_EXPECT_OPERAND;

    std::cout << "\nConfiguration trace  (state, unread input, stack)\n";
    std::cout << "----------------------------------------\n";
    std::cout << " 0: (" << state_name(state) << ", " << s << ", "
              << stack_to_string(stack) << ")\n";

    for (i = 0; i <= s.size(); i++) {
        int symbol = (i < s.size()) ? symbol_index(s[i]) : SYM_END;
        Transition step;

        if (symbol < 0) {
            std::cout << " " << (i + 1) << ": (qD, --, " << stack_to_string(stack)
                      << ")   '" << s[i] << "' is not in Sigma\n";
            std::cout << "\nRESULT: REJECT\n";
            return;
        }

        step = delta[state][symbol][stack.back()];

        apply_action(stack, step.action);
        state = step.nextState;

        std::cout << " " << (i + 1) << ": (" << state_name(state) << ", ";
        if (i < s.size())
            std::cout << s.substr(i + 1);
        else
            std::cout << "e";
        std::cout << ", " << stack_to_string(stack) << ")"
                  << "   on '" << symbol_name(symbol) << "'\n";

        if (state == ST_DEAD) {
            std::cout << "\nRESULT: REJECT\n";
            return;
        }
    }

    if (state == ST_EXPECT_OPERATOR && stack.size() == 1)
        std::cout << "\nRESULT: ACCEPT\n";
    else
        std::cout << "\nRESULT: REJECT\n";
}

/* ===========================================================================
 *  4.  THE BUILT-IN TEST SUITE
 *
 *  Every case carries the tag of the rule of the specification that it checks,
 *  so the summary printed at the end shows that each rule of the problem
 *  statement is covered by at least one passing test.
 * ======================================================================== */

struct TestCase {
    const char* input;
    bool        expected;
    const char* rule;      /* the rule of the specification being checked */
    const char* note;
};

static const TestCase TESTS[] = {
    /* --- the ten examples of the problem statement ------------------------- */
    { "3+4",                   true,  "R6", "a number, an operator, a number" },
    { "(3+4)*(2-(1+5))",       true,  "R1", "the example: balanced nested parentheses" },
    { "-5+3",                  true,  "R4", "the example: unary minus at the start" },
    { "0+1",                   true,  "R5", "the example: a single 0 is allowed" },
    { "3++4",                  false, "R3", "the example: two operators consecutively" },
    { "+3",                    false, "R2", "the example: operator at the start" },
    { "3+",                    false, "R2", "the example: operator at the end" },
    { "(3+4",                  false, "R1", "the example: unbalanced parentheses" },
    { "007+1",                 false, "R5", "the example: leading zeros" },
    { "3*(/2)",                false, "R3", "the example: an operator after ( with no operand" },
    /* --- rule 1: parentheses are balanced and properly nested -------------- */
    { "(3+4)",                 true,  "R1", "one balanced group" },
    { "((1+2)*(3+4))",         true,  "R1", "two levels of nesting" },
    { "((1))",                 true,  "R1", "a group around a group" },
    { "3+4)",                  false, "R1", "one closing parenthesis too many" },
    { "(()))",                 false, "R1", "an empty group and a stray )" },
    { ")(",                    false, "R1", "a closing parenthesis with no opener" },
    { "(1+2))",                false, "R1", "the second group is never closed" },
    { "()",                    false, "R1", "an empty group has no operand" },
    /* --- rule 2: no operator at the start or at the end -------------------- */
    { "*2",                    false, "R2", "an operator at the start" },
    { "/2",                    false, "R2", "an operator at the start" },
    { "3*",                    false, "R2", "an operator at the end" },
    { "3/",                    false, "R2", "an operator at the end" },
    { "-",                     false, "R2", "a lone minus is not an operand" },
    /* --- rule 3: no two operators in a row --------------------------------- */
    { "3**4",                  false, "R3", "two multiplication signs" },
    { "3//4",                  false, "R3", "two division signs" },
    { "3+-4",                  true,  "R3", "a unary minus may follow an operator" },
    { "3--4",                  true,  "R3", "a unary minus may follow an operator" },
    { "3+*4",                  false, "R3", "the second operator is not a minus" },
    { "(*2)",                  false, "R3", "an operator straight after an opener" },
    /* --- rule 4: the unary minus is only at the start, after ( or after an op */
    { "5*-3",                  true,  "R4", "a unary minus after an operator" },
    { "3*(-2)",                true,  "R4", "a unary minus after an opening parenthesis" },
    { "-(3+4)",                true,  "R4", "a unary minus on a group at the start" },
    { "5/-3",                  true,  "R4", "a unary minus after a division sign" },
    { "(-3)",                  true,  "R4", "a unary minus inside a group" },
    /* --- rule 5: no leading zeros ------------------------------------------ */
    { "0",                     true,  "R5", "the single digit 0 is allowed" },
    { "0-5",                   true,  "R5", "a zero followed by an operator" },
    { "01",                    false, "R5", "a leading zero" },
    { "00",                    false, "R5", "leading zeros" },
    { "000",                   false, "R5", "leading zeros" },
    { "10",                    true,  "R5", "a zero inside a number is fine" },
    { "100+20",                true,  "R5", "zeros inside numbers are fine" },
    /* --- rule 6: a well-formed expression ---------------------------------- */
    { "42",                    true,  "R6", "a single number is a complete expression" },
    { "10/2",                  true,  "R6", "a two-digit operand" },
    { "1234567890",            true,  "R6", "a long number" },
    { "1+2+3+4+5",             true,  "R6", "a chain of binary operators" },
    { "1/0",                   true,  "R6", "only the syntax is checked, not the value" },
    { "2(3)",                  false, "R6", "implicit multiplication is not allowed" },
    { "(1)(2)",                false, "R6", "implicit multiplication is not allowed" },
    /* --- boundary conditions and characters outside the alphabet ----------- */
    { "",                      false, "R7", "the empty string" },
    { "5",                     true,  "R7", "the shortest valid expression" },
    { "3 4",                   false, "RX", "a space is not in Sigma" },
    { "1a",                    false, "RX", "a letter is not in Sigma" },
    { "3.5",                   false, "RX", "a decimal point is not in Sigma" }
};

/* The rules of the specification, in the order in which the report lists them. */
static const char* const RULE_NAMES[] = {
    "R1  parentheses are balanced and properly nested",
    "R2  no operator at the start or at the end of the expression",
    "R3  no two operators in a row",
    "R4  a unary minus only at the start, after ( or after an operator",
    "R5  no digit sequence with a leading zero unless it is exactly 0",
    "R6  a well-formed expression: operands and operators alternate",
    "R7  a boundary condition of the specification",
    "RX  a character that is not in the alphabet"
};

static const int NUM_RULES = static_cast<int>(sizeof(RULE_NAMES) / sizeof(RULE_NAMES[0]));
static const int NUM_TESTS = static_cast<int>(sizeof(TESTS) / sizeof(TESTS[0]));

/* Run the built-in suite and print a report.  Returns the number of failures. */
static int run_tests()
{
    int failures = 0;
    int i;
    int rule;

    std::cout << "\n #  Rule  Input                        Expected  Observed"
              << "   Result  Note\n";
    std::cout << "----------------------------------------------------------------"
                 "----------------------------------------------------\n";

    for (i = 0; i < NUM_TESTS; i++) {
        bool observed = accepts(TESTS[i].input);
        bool passed = (observed == TESTS[i].expected);
        std::string text(TESTS[i].input);

        if (!passed)
            failures++;

        std::cout << " " << (i + 1);
        if (i + 1 < 10) std::cout << " ";
        std::cout << "  " << TESTS[i].rule << "   \"" << text << "\"";
        for (std::size_t pad = text.size(); pad < 24; pad++)
            std::cout << ' ';
        std::cout << (TESTS[i].expected ? "ACCEPT  " : "REJECT  ");
        std::cout << (observed ? "ACCEPT  " : "REJECT  ");
        std::cout << (passed ? "PASS" : "FAIL");
        std::cout << "     " << TESTS[i].note << "\n";
    }

    std::cout << "----------------------------------------------------------------"
                 "----------------------------------------------------\n";

    /* Summary: one line per rule of the specification, so that it is visible
     * at a glance that every rule is covered by passing tests. */
    std::cout << "\nCoverage of the rules of the specification\n";
    std::cout << "----------------------------------------------------------------"
                 "--------------------\n";

    for (rule = 0; rule < NUM_RULES; rule++) {
        int total = 0, ok = 0;

        for (i = 0; i < NUM_TESTS; i++) {
            if (std::string(TESTS[i].rule) != std::string(RULE_NAMES[rule]).substr(0, 2))
                continue;
            total++;
            if (accepts(TESTS[i].input) == TESTS[i].expected)
                ok++;
        }

        std::cout << "  " << RULE_NAMES[rule] << " ";
        for (std::size_t pad = std::string(RULE_NAMES[rule]).size() + 1; pad < 60; pad++)
            std::cout << ' ';
        std::cout << ok << " / " << total << " passed\n";
    }

    std::cout << "----------------------------------------------------------------"
                 "--------------------\n";
    std::cout << (NUM_TESTS - failures) << " / " << NUM_TESTS
              << " test cases passed.\n";

    return failures;
}

/* ===========================================================================
 *  5.  THE CONSOLE FRONT END
 * ======================================================================== */

static void print_banner()
{
    std::cout << "\n=====================================================\n";
    std::cout << "  AUTOMATA - Arithmetic Expression Validator\n";
    std::cout << "  Deterministic Push-Down Automaton (DPDA)\n";
    std::cout << "  M = (Q, Sigma, Gamma, delta, q0, Z0, F)\n";
    std::cout << "=====================================================\n";
}

static void print_menu()
{
    std::cout << "\n1. Check an expression\n";
    std::cout << "2. Show the step-by-step PDA trace\n";
    std::cout << "3. Run the built-in test suite\n";
    std::cout << "4. Exit\n";
    std::cout << "Choose an option: ";
}

int main(int argc, char* argv[])
{
    int choice;

    /* Command line mode:  program.exe "3+4"  */
    if (argc > 1) {
        if (std::string(argv[1]) == "--test") {
            int failures = run_tests();
            return failures == 0 ? 0 : 1;
        }
        std::cout << "\"" << argv[1] << "\" -> "
                  << (accepts(argv[1]) ? "ACCEPT" : "REJECT") << "\n";
        return 0;
    }

    /* Interactive mode. */
    print_banner();

    do {
        print_menu();

        if (!(std::cin >> choice)) {          /* end of input */
            std::cout << "\n";
            return 0;
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch (choice) {
        case 1: {
            std::string expression;
            std::cout << "\nEnter an arithmetic expression: ";
            std::getline(std::cin, expression);
            std::cout << "Result: "
                      << (accepts(expression) ? "ACCEPTED" : "REJECTED")
                      << "\n";
            break;
        }
        case 2: {
            std::string expression;
            std::cout << "\nEnter an arithmetic expression: ";
            std::getline(std::cin, expression);
            trace(expression);
            break;
        }
        case 3:
            run_tests();
            break;
        case 4:
            std::cout << "\nProgram terminated.\n";
            break;
        default:
            std::cout << "\nInvalid choice. Please choose 1, 2, 3 or 4.\n";
        }
    } while (choice != 4);

    return 0;
}
