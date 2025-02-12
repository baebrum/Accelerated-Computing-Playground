#include <iostream>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <random>

const int DEFAULT_N = 32;
const int DEFAULT_THREADS = 2;
const int MAX_RANDOM_VALUE = 1000;

// Node class definition
class Node {
public:
    Node();
    ~Node();

    int value;
    Node* next;
    Node* prev;
    omp_lock_t lock;
};

Node::Node() {}
Node::~Node() {}

// Function Prototypes
void printUsage();
bool isPowerOfTwo(int n);
void cleanLinkedList(Node* head, bool debug_mode);
Node* mergeSortedLists(Node* list1, Node* list2);
void handleCommandLineArguments(int argc, char** argv, int& N, int& num_threads, bool& debug_mode);
void processLinkedListWorkload(int N, int num_threads, bool debug_mode);

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

// Function Definitions

void printUsage() {
    std::cout << "Usage: ./main [-n N] [-t THREADS] [-d]\n";
    std::cout << "  -n N        : Set the size of the linked list (must be a power of 2). Default is N = 32.\n";
    std::cout << "  -t THREADS  : Set the number of threads for OpenMP. Default is THREADS = 2.\n";
    std::cout << "  -d          : Enable debug mode to print debug information.\n";
    std::cout << "\n";
    std::cout << "Examples:\n";
    std::cout << "  ./main\n";
    std::cout << "  ./main -n 64\n";
    std::cout << "  ./main -t 8\n";
    std::cout << "  ./main -n 64 -t 8\n";
    std::cout << "  ./main -d\n";
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
        delete currentNode;
        currentNode = nextNode;
    }

    if (debug_mode) {
        std::cout << "\nCleaned up " << nodeCount << " nodes.\n";
    }
}

Node* mergeSortedLists(Node* list1, Node* list2) {
    if (!list1) return list2;
    if (!list2) return list1;

    Node* head = (list1->value <= list2->value) ? list1 : list2;
    Node* current = head;

    if (head == list1) list1 = list1->next;
    else list2 = list2->next;

    while (list1 && list2) {
        if (list1->value <= list2->value) {
            current->next = list1;
            list1->prev = current;
            list1 = list1->next;
        }
        else {
            current->next = list2;
            list2->prev = current;
            list2 = list2->next;
        }
        current = current->next;
    }

    if (list1) current->next = list1;
    if (list2) current->next = list2;

    return head;
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

    int* values = new int[N];

    // Parallel generation of random values
#pragma omp parallel for
    for (int k = 0; k < N; k++) {
        values[k] = dis(gen);
    }

    // Structure to store thread-specific data
    struct ThreadData {
        Node* head;
        Node* tail;
        int inserted_nodes;
    };

    ThreadData* thread_data = new ThreadData[num_threads]();

    double start_time = omp_get_wtime();

    // Parallel sorting and insertion of nodes
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        thread_data[tid].head = nullptr;
        thread_data[tid].tail = nullptr;

#pragma omp for
        for (int k = 0; k < N; k++) {
            Node* newNode = new Node();
            newNode->value = values[k];
            newNode->next = nullptr;
            newNode->prev = nullptr;

            // Insert the node into the local sorted list
            if (!thread_data[tid].head || thread_data[tid].head->value >= newNode->value) {
                newNode->next = thread_data[tid].head;
                if (thread_data[tid].head) {
                    thread_data[tid].head->prev = newNode;
                }
                thread_data[tid].head = newNode;
            }
            else {
                Node* p = thread_data[tid].head;
                while (p->next && p->next->value < newNode->value) {
                    p = p->next;
                }
                newNode->next = p->next;
                if (p->next) {
                    p->next->prev = newNode;
                }
                p->next = newNode;
                newNode->prev = p;
            }
            thread_data[tid].inserted_nodes++;
        }
    }

    // Merging the local sorted lists into the final sorted list
    Node* head = thread_data[0].head;
    for (int i = 1; i < num_threads; i++) {
        head = mergeSortedLists(head, thread_data[i].head);
    }

    double run_time = omp_get_wtime() - start_time;

    cleanLinkedList(head, debug_mode);
    delete[] values;
    delete[] thread_data;

    if (debug_mode) {
        std::cout << "Total nodes inserted: " << N << "\n";
    }
    std::cout << "Workload: " << run_time << " seconds\n";
}
