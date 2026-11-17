"""
Problem 8: Learning and Visualizing DFA/NFA Models from Training Data
RPNI-based DFA learning demonstration.

Example target language: binary strings ending with "01".
Run in Google Colab or locally with Python.
"""

# If running in Google Colab, uncomment this line:
# !pip install numpy pandas matplotlib networkx scikit-learn

import copy
import time
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. DATASET GENERATION
# ============================================================

ALPHABET = ["0", "1"]
MAX_TRAIN_LENGTH = 6


def target_language(string):
    """Ground-truth label: 1 if string ends with '01', otherwise 0."""
    return string.endswith("01")


def generate_dataset(alphabet=ALPHABET, max_length=MAX_TRAIN_LENGTH):
    positive = []
    negative = []

    for length in range(1, max_length + 1):
        for symbols in product(alphabet, repeat=length):
            string = "".join(symbols)
            if target_language(string):
                positive.append(string)
            else:
                negative.append(string)

    return positive, negative


positive_examples, negative_examples = generate_dataset()

dataset = pd.DataFrame(
    [(s, 1) for s in positive_examples] +
    [(s, 0) for s in negative_examples],
    columns=["String", "Label"]
)

print("DATASET SUMMARY")
print("----------------")
print("Positive examples:", len(positive_examples))
print("Negative examples:", len(negative_examples))
print("Total examples:", len(dataset))
print(dataset.head(10).to_string(index=False))


# ============================================================
# 2. PREFIX TREE ACCEPTOR (PTA)
# ============================================================

def build_pta(positive_strings):
    """
    Build a Prefix Tree Acceptor from positive examples.
    State 0 is the initial state.
    """
    pta = {
        0: {
            "final": False,
            "transitions": {}
        }
    }

    next_state = 1

    for string in positive_strings:
        current = 0

        for symbol in string:
            if symbol not in pta[current]["transitions"]:
                pta[next_state] = {
                    "final": False,
                    "transitions": {}
                }
                pta[current]["transitions"][symbol] = next_state
                next_state += 1

            current = pta[current]["transitions"][symbol]

        pta[current]["final"] = True

    return pta


# ============================================================
# 3. DFA RECOGNITION
# ============================================================

def recognize(automaton, string, initial_state=0):
    """Return True if the automaton accepts string; otherwise False."""
    current = initial_state

    for symbol in string:
        if symbol not in automaton[current]["transitions"]:
            return False
        current = automaton[current]["transitions"][symbol]

    return automaton[current]["final"]


# ============================================================
# 4. RPNI STATE MERGING
# ============================================================

def find(parent, state):
    """Find the representative of a disjoint-set state."""
    if parent[state] != state:
        parent[state] = find(parent, parent[state])
    return parent[state]


def merge_states(parent, pta, state1, state2, visited=None):
    """
    Recursively merge state2 into state1.

    The merge fails if an accepting state would be merged with a
    non-accepting state. Conflicting outgoing transitions are merged
    recursively. This function modifies the candidate parent array;
    callers should pass a copy when testing a tentative merge.
    """
    if visited is None:
        visited = set()

    r1 = find(parent, state1)
    r2 = find(parent, state2)

    if r1 == r2:
        return True

    pair = tuple(sorted((r1, r2)))
    if pair in visited:
        return True
    visited.add(pair)

    members1 = [s for s in pta if find(parent, s) == r1]
    members2 = [s for s in pta if find(parent, s) == r2]

    final1 = any(pta[s]["final"] for s in members1)
    final2 = any(pta[s]["final"] for s in members2)

    if final1 != final2:
        return False

    # Gather all outgoing transitions before joining the two classes.
    combined_transitions = {}
    for state in members1 + members2:
        for symbol, destination in pta[state]["transitions"].items():
            combined_transitions.setdefault(symbol, []).append(destination)

    # Join r2 into r1.
    parent[r2] = r1

    # For each symbol, all destinations must become one state.
    for destinations in combined_transitions.values():
        first_destination = destinations[0]

        for other_destination in destinations[1:]:
            if not merge_states(
                parent,
                pta,
                first_destination,
                other_destination,
                visited
            ):
                return False

            first_destination = find(parent, first_destination)

    return True


def build_merged_dfa(parent, pta):
    """Convert the merged PTA equivalence classes into a reachable DFA."""
    groups = {}

    for state in pta:
        representative = find(parent, state)
        groups.setdefault(representative, []).append(state)

    transitions = {}
    final_states = {}

    for representative, members in groups.items():
        final_states[representative] = any(
            pta[state]["final"] for state in members
        )
        transitions[representative] = {}

        for state in members:
            for symbol, destination in pta[state]["transitions"].items():
                destination = find(parent, destination)

                if symbol in transitions[representative]:
                    existing = transitions[representative][symbol]
                    if find(parent, existing) != destination:
                        raise ValueError(
                            "Inconsistent transitions after state merging."
                        )
                else:
                    transitions[representative][symbol] = destination

    initial_state = find(parent, 0)
    reachable = set()
    stack = [initial_state]

    while stack:
        state = stack.pop()
        if state in reachable:
            continue

        reachable.add(state)

        for destination in transitions[state].values():
            stack.append(destination)

    dfa = {}
    for state in reachable:
        dfa[state] = {
            "final": final_states[state],
            "transitions": {
                symbol: destination
                for symbol, destination in transitions[state].items()
                if destination in reachable
            }
        }

    return dfa


def rpni_learning(positive, negative, pta):
    """
    Basic blue-fringe RPNI.

    Starts with the PTA of positive examples. It attempts to merge each
    blue state into each red state. A tentative merge is retained only
    if it does not cause any supplied negative example to be accepted.
    """
    if any(string == "" for string in negative):
        # The empty string is rejected by the example target language.
        pass

    parent = list(range(len(pta)))
    red = {0}

    # A PTA built from positive examples should not already accept negatives.
    initial_dfa = build_merged_dfa(parent, pta)
    if any(recognize(initial_dfa, s) for s in negative):
        raise ValueError(
            "The positive and negative examples are inconsistent: "
            "the initial PTA already accepts a negative example."
        )

    while True:
        red_representatives = {
            find(parent, state) for state in red
        }

        current_dfa = build_merged_dfa(parent, pta)

        blue = set()
        for red_state in red_representatives:
            for destination in current_dfa[red_state]["transitions"].values():
                destination = find(parent, destination)
                if destination not in red_representatives:
                    blue.add(destination)

        if not blue:
            break

        blue_state = min(blue)
        accepted_merge = False

        for red_state in sorted(red_representatives):
            candidate_parent = parent.copy()

            successful_merge = merge_states(
                candidate_parent,
                pta,
                red_state,
                blue_state
            )

            if not successful_merge:
                continue

            candidate_dfa = build_merged_dfa(candidate_parent, pta)

            if not any(
                recognize(candidate_dfa, negative_string)
                for negative_string in negative
            ):
                parent = candidate_parent
                accepted_merge = True
                break

        if not accepted_merge:
            red.add(blue_state)

    return build_merged_dfa(parent, pta)


# ============================================================
# 5. RUN RPNI
# ============================================================

pta = build_pta(positive_examples)

start_time = time.perf_counter()
learned_dfa = rpni_learning(
    positive_examples,
    negative_examples,
    pta
)
execution_time = time.perf_counter() - start_time

print("\nRPNI LEARNING RESULT")
print("--------------------")
print("Initial PTA states:", len(pta))
print("Learned DFA states:", len(learned_dfa))
print("Execution time (seconds):", round(execution_time, 6))

for state, details in sorted(learned_dfa.items()):
    print(
        f"q{state}: final={details['final']}, "
        f"transitions={details['transitions']}"
    )


# ============================================================
# 6. VISUALIZE THE LEARNED DFA
# ============================================================

def visualize_dfa(dfa):
    graph = nx.DiGraph()

    for state in dfa:
        graph.add_node(state)

    # Combine labels when multiple symbols connect the same pair of states.
    for state, details in dfa.items():
        for symbol, destination in details["transitions"].items():
            if graph.has_edge(state, destination):
                graph[state][destination]["label"] += ", " + symbol
            else:
                graph.add_edge(state, destination, label=symbol)

    pos = nx.spring_layout(graph, seed=42)

    plt.figure(figsize=(9, 6))
    nx.draw_networkx_edges(
        graph, pos, arrows=True, arrowsize=20,
        connectionstyle="arc3,rad=0.08"
    )

    non_final = [
        state for state in dfa if not dfa[state]["final"]
    ]
    final = [
        state for state in dfa if dfa[state]["final"]
    ]

    nx.draw_networkx_nodes(
        graph, pos, nodelist=non_final,
        node_size=1700, node_color="lightblue"
    )
    nx.draw_networkx_nodes(
        graph, pos, nodelist=final,
        node_size=1700, node_color="lightgreen",
        node_shape="s"
    )

    nx.draw_networkx_labels(
        graph, pos,
        labels={state: f"q{state}" for state in graph.nodes()},
        font_size=12
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_size=11
    )

    # Draw an incoming arrow to the initial state.
    initial_state = 0
    x, y = pos[initial_state]
    plt.annotate(
        "",
        xy=(x, y),
        xytext=(x - 0.25, y + 0.25),
        arrowprops=dict(arrowstyle="->", lw=1.5)
    )

    plt.title("DFA Learned Using RPNI")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


visualize_dfa(learned_dfa)


# ============================================================
# 7. TEST USER-DEFINED STRINGS
# ============================================================

def test_strings(strings, dfa):
    print("\nSTRING RECOGNITION")
    print("------------------")

    for string in strings:
        result = recognize(dfa, string)
        print(
            repr(string),
            "->",
            "ACCEPTED" if result else "REJECTED"
        )


sample_test_strings = [
    "01", "101", "1001", "1101",
    "10", "11", "000", "1010",
    "11101", "10000"
]

test_strings(sample_test_strings, learned_dfa)


# ============================================================
# 8. TRAINING ACCURACY
# ============================================================

all_training_strings = positive_examples + negative_examples
true_training_labels = (
    [1] * len(positive_examples) +
    [0] * len(negative_examples)
)

predicted_training_labels = [
    int(recognize(learned_dfa, s))
    for s in all_training_strings
]

training_accuracy = accuracy_score(
    true_training_labels,
    predicted_training_labels
)

print("\nTRAINING EVALUATION")
print("-------------------")
print("Training accuracy:", round(training_accuracy * 100, 2), "%")
print(
    classification_report(
        true_training_labels,
        predicted_training_labels,
        zero_division=0
    )
)


# ============================================================
# 9. GENERALIZATION ON UNSEEN LONGER STRINGS
# ============================================================

unseen_strings = []

for length in range(7, 10):
    for symbols in product(ALPHABET, repeat=length):
        unseen_strings.append("".join(symbols))

true_unseen_labels = [
    int(target_language(s)) for s in unseen_strings
]

predicted_unseen_labels = [
    int(recognize(learned_dfa, s)) for s in unseen_strings
]

generalization_accuracy = accuracy_score(
    true_unseen_labels,
    predicted_unseen_labels
)

print("\nGENERALIZATION EVALUATION")
print("-------------------------")
print("Unseen strings:", len(unseen_strings))
print(
    "Generalization accuracy:",
    round(generalization_accuracy * 100, 2),
    "%"
)


# ============================================================
# 10. FINAL PERFORMANCE REPORT
# ============================================================

report = pd.DataFrame({
    "Metric": [
        "Positive examples",
        "Negative examples",
        "Total examples",
        "Initial PTA states",
        "Learned DFA states",
        "Training accuracy (%)",
        "Unseen test accuracy (%)",
        "RPNI execution time (seconds)"
    ],
    "Value": [
        len(positive_examples),
        len(negative_examples),
        len(dataset),
        len(pta),
        len(learned_dfa),
        round(training_accuracy * 100, 2),
        round(generalization_accuracy * 100, 2),
        round(execution_time, 6)
    ]
})

print("\nFINAL PROJECT REPORT")
print("--------------------")
print(report.to_string(index=False))


# ============================================================
# 11. OPTIONAL: SAVE OUTPUTS
# ============================================================

dataset.to_csv("training_dataset.csv", index=False)
report.to_csv("performance_report.csv", index=False)

print("\nSaved training_dataset.csv and performance_report.csv")
