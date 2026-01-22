import json
import statistics

input_path = "results/retrieved_knowledge/metaqa-1hop/gpt-5-mini/test/retrieved_knowledge.jsonl"
output_path = "results/retrieved_knowledge/metaqa-1hop/gpt-5-mini/test/retrieved_knowledge_eval.jsonl"

total = 0
hit_count = 0
total_retrieve_count = 0
retrieve_counts = []

# Process each line of the input file
with open(input_path, "r", encoding="utf-8") as infile, \
        open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        retrieved_knowledge = data.get("retrieved_knowledge", "")
        a_entities = data.get("a_entity", [])

        # Check if any a_entity is mentioned in retrieved_knowledge
        hit = 0
        for entity in a_entities:
            if entity in retrieved_knowledge:
                hit = 1
                break

        data["hit"] = hit
        hit_count += hit
        total += 1

        retrieve_count = data.get("retrieve_count", 0)
        total_retrieve_count += retrieve_count
        if hit:
            retrieve_counts.append(retrieve_count)

        # Write the updated record to output file
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write("\n")

# Compute and display average hit rate
if total > 0:
    avg_hit = hit_count / total
    avg_retrieve_count = total_retrieve_count / total
    median_retrieve_count = statistics.median(retrieve_counts)

    count_1 = sum(1 for x in retrieve_counts if x == 1)
    count_2_3 = sum(1 for x in retrieve_counts if x in [2, 3])
    count_4_5 = sum(1 for x in retrieve_counts if x in [4, 5])
    count_gt5 = sum(1 for x in retrieve_counts if x > 5)
    print(f"Average hit rate: {avg_hit:.4f} ({hit_count}/{total})")
    print(f"Average retrieve count: {avg_retrieve_count:.4f} ({total_retrieve_count}/{total})")
    print(f"Median retrieve count: {median_retrieve_count}")
    print("\nRetrieve Count Distribution:")
    print(f"retrieve_count == 1 : {count_1} ({count_1 / total:.2%})")
    print(f"retrieve_count == 2 or 3 : {count_2_3} ({count_2_3 / total:.2%})")
    print(f"retrieve_count == 4 or 5 : {count_4_5} ({count_4_5 / total:.2%})")
    print(f"retrieve_count > 5 : {count_gt5} ({count_gt5 / total:.2%})")
else:
    print("No records found in the input file.")
