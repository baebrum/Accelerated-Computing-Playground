// avoid using this file with multiple threads as it is inefficient
// multiple threads is only used to show inefficiency and resource
// contention

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <random>

const int DEFAULT_N = 32;
const int DEFAULT_THREADS = 2;
const int MAX_RANDOM_VALUE = 1000;

struct Node {
    int value;
    struct Node* next;
    struct Node* prev;
    omp_lock_t lock;
};

// Function Prototypes
void printUsage();
bool isPowerOfTwo(int n);
void handleCommandLineArguments(int argc, char** argv, int& N, int& num_threads, bool& debug_mode);
void processLinkedListWorkload(int N, int num_threads, bool debug_mode);
void cleanLinkedList(Node* head, bool debug_mode);

int main(int argc, char** argv) {
    int N = DEFAULT_N;
    int num_threads = DEFAULT_THREADS;
    bool debug_mode = false;

    handleCommandLineArguments(argc, argv, N, num_threads, debug_mode);

    std::cout << "Using N = " << N << " and num_threads = " << num_threads << "\n";
    if (debug_mode) {
        std::cout << "Debug mode enabled.\n";
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    processLinkedListWorkload(N, num_threads, debug_mode);

    return 0;
}

void printUsage() {
    std::cout << "Usage: ./serial [-n N] [-t THREADS] [-d]\n";
    std::cout << "  -n N        : Set the size of the linked list (must be a power of 2). Default is N = 32.\n";
    std::cout << "  -t THREADS  : Set the number of threads for OpenMP. Default is THREADS = 2.\n";
    std::cout << "  -d          : Enable debug mode to print debug information.\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  ./serial\n";
    std::cout << "  ./serial -n 64\n";
    std::cout << "  ./serial -t 8\n";
    std::cout << "  ./serial -n 64 -t 8\n";
    std::cout << "  ./serial -d\n";
}

bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

void cleanLinkedList(Node* head, bool debug_mode) {
    Node* currentNode = head;
    int nodeCount = 0;

    while (currentNode != nullptr) {
        nodeCount++;
        if (debug_mode) {
            std::printf("%03d, ", currentNode->value);
        }
        Node* nextNode = currentNode->next;
        delete currentNode;  // Using delete instead of free
        currentNode = nextNode;
    }

    if (debug_mode) {
        std::cout << "\nCleaned up " << nodeCount << " nodes.\n";
    }
}

void handleCommandLineArguments(int argc, char** argv, int& N, int& num_threads, bool& debug_mode) {
    if (argc == 1) {
        std::cout << "Using default N = " << DEFAULT_N << " and num_threads = " << DEFAULT_THREADS << ".\n";
        return;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) {
                N = std::atoi(argv[++i]);
                if (!isPowerOfTwo(N)) {
                    std::cerr << "Error: N must be a power of 2. Defaulting to N = " << DEFAULT_N << ".\n";
                    N = DEFAULT_N;
                }
            }
            else {
                std::cerr << "Error: Missing value for -n.\n";
                printUsage();
                std::exit(1);
            }
        }
        else if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) {
                num_threads = std::atoi(argv[++i]);
                if (num_threads <= 0) {
                    std::cerr << "Error: Invalid number of threads. Defaulting to THREADS = " << DEFAULT_THREADS << ".\n";
                    num_threads = DEFAULT_THREADS;
                }
            }
            else {
                std::cerr << "Error: Missing value for -t.\n";
                printUsage();
                std::exit(1);
            }
        }
        else if (strcmp(argv[i], "-d") == 0) {
            debug_mode = true;
        }
        else {
            std::cerr << "Error: Unrecognized argument " << argv[i] << "\n";
            printUsage();
            std::exit(1);
        }
    }
}

void processLinkedListWorkload(int N, int num_threads, bool debug_mode) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, MAX_RANDOM_VALUE);

    Node* head = nullptr;
    Node* p, * prev = nullptr; // working pointers
    int value;
    int k;
    Node* newNode;
    // create first Node
    head = new Node();
    head->next = nullptr;
    head->prev = nullptr;
    omp_init_lock(&head->lock);
    omp_lock_t lock;
    double start_time = omp_get_wtime();

#pragma omp parallel for private(k, value, newNode, p, prev)
    for (k = 1; k <= N; k++) {
        omp_set_lock(&lock);
        p = head->next;
        prev = head;
        value = dis(gen);
        while (p != nullptr) {
            if (p->value >= value)
                break;
            prev = p;
            p = p->next;
        }
        newNode = new Node();
        newNode->value = value;
        newNode->next = p;
        newNode->prev = prev;
        newNode->prev->next = newNode;

        if (p) {
            p->prev = newNode;
        }
        omp_unset_lock(&lock);
    }

    double run_time = omp_get_wtime() - start_time;

    cleanLinkedList(head, debug_mode);

    if (debug_mode) {
        std::cout << "Total nodes inserted: " << N << "\n";
    }
    std::cout << "Workload: " << run_time << " seconds\n";
}
