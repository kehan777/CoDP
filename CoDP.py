from model import CoDP
from LigandMPNN.package import MPNNModel
import argparse
import os
import pickle
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF3 based optimizer')
    parser.add_argument('--input_file_pkl', type=str, required=True,
                       help='input file path')
    parser.add_argument('--output_file_pkl', type=str, required=True,
                       help='Directory for output')
    parser.add_argument('--num_seqs', type=int, default=8,
                       help='Number of proteinMPNN or LigadnMPNN+proteinMPNN seqs to perform self consistency')
    parser.add_argument('--pdb_file', type=str, required=False,
                       help='input pdb path')
    return parser.parse_args()

def run_purempnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,weights_str,output_dir,bias_AA,symmetry_residues):  
    

    print(f"{mpnn_config_dict['model_name']} design")
    output_dir_mpnn =os.path.join(output_dir,f"{mpnn_config_dict['model_name']}")
    sequences_stack,pdb_path_stack=mpnn_model.single_protein_mpnn_design(scaffold_path=scaffold_path, 
                                      output_dir_mpnn=output_dir_mpnn, 
                                      numbers_seqs=mpnn_config_dict['num_seqs'],   
                                      chains_to_design="",
                                      fixed_res=pocket_res_to_fix, 
                                      redesigned_residues="", 
                                      symmetry_residues=symmetry_residues,
                                      input_bias_AA = bias_AA,
                                      weights_str=weights_str,)

    return  sequences_stack,pdb_path_stack

import time
def run_selection_process(sequences, packed_paths, evaluator, scaffold_path, num_seqs,batch_size):
    """
    Performs multi-round tournament-style selection based on prediction scores.

    Args:
        sequences (list): List of generated amino acid sequences.
        packed_paths (list): List of PDB file paths corresponding to the sequences.
        evaluator (object): Evaluator object used for predicting scores.
        scaffold_path (str): Path to the scaffold PDB file.
        mpnn_config_dict (dict): Dictionary containing configuration, including 'num_seqs'.

    Returns:
        list: Filtered list of (sequence, packed_path, score) tuples, sorted in descending order by score.
    """
    current_data = [(seq, packed) for seq, packed in zip(sequences, packed_paths)]
    #print(current_data)
    # target number of sequences to retain
    target_num_seqs = num_seqs # extract from mpnn_config_dict
    start_time = time.time()
    print(f"init counts: {len(current_data)}")
    print(f"target counts: {target_num_seqs}")
    # make sure target_num_seqs at least 1
    if target_num_seqs == 0:
        target_num_seqs = 1 
    while len(current_data) > target_num_seqs:
        # max number of sequences to process in one batch
        all_scores_for_round = []
        
        # in batch to predict
        for i in range(0, len(current_data), batch_size):
            batch_sequences = [item[0] for item in current_data[i : i + batch_size]]
            batch_packed_paths = [item[1] for item in current_data[i : i + batch_size]]

            batch_scores = evaluator.predict(batch_sequences, scaffold_path)

            for j, (seq, packed) in enumerate(zip(batch_sequences, batch_packed_paths)):
                all_scores_for_round.append((seq, packed, batch_scores[j]))
        
        # Sort all sequences in the current round by score (descending)
        all_scores_for_round.sort(key=lambda x: -x[2])  # x[2] is the score
        
        # Select top half sequences for the next round
        # For odd numbers, take (N+1)/2 to ensure at least half are kept
        next_round_count = max(target_num_seqs, (len(all_scores_for_round) + 1) // 2)
        
        # Ensure we don't select more sequences than available
        # The loop condition already prevents current round from being smaller than target
        
        current_data = [(item[0], item[1]) for item in all_scores_for_round[:next_round_count]]
        
        print(f"finial count: {len(current_data)}")
        if len(current_data) == 0:  # Prevent infinite loops and handle empty lists
            print("Warning: Filtered sequence list is empty, terminating early.")
            break
    
        # Final results need to follow the complete interaction_data format: (seq, packed, 0, 0, 0, score)
        # Here 'score' should be the final evaluation score, other fields are filled with 0
        final_results = []
        for seq, packed in current_data:
            # Find the final score for this sequence
            # In theory, all_scores_for_round stores the scores from the final round
            # To maintain compatibility with updated_sequences format, we need to populate the score field
            # If evaluator.predict returns the final score, it should be used here

            # Look up the final score for this sequence (this requires tracking scores across rounds)
            # For simplicity, we assume you need both final sequences and their corresponding scores
            # We'll search for the score in the final round's results (all_scores_for_round)

            # Find the current sequence's score from the final round
            final_score = 0
            for item in all_scores_for_round:  # Ensure all_scores_for_round contains complete final round data
                if item[0] == seq and item[1] == packed:
                    final_score = item[2]
                    break
        
            final_results.append((seq, packed, final_score))

    final_results.sort(key=lambda x: -x[2])

    end_time = time.time()
    elapsed_time = end_time - start_time
    return final_results



def main():
    args = parse_arguments()
    checkpoints_to_run = os.path.dirname(os.path.abspath(__file__))+ "/ckpt/epoch_1_without_esm2.pth"
    ##! need change
    esm_name = "facebook/esm2_t33_650M_UR50D" # 
    evaluator = CoDP(checkpoints_to_run,esm_name)

    with open(args.input_file_pkl, "rb") as f:
        input_file_pkl = pickle.load(f)
    [sequences, packed_paths]= input_file_pkl
    print(sequences,packed_paths)
    if evaluator :
        num_seqs = args.num_seqs
        batchsizes = [8, 4, 2]
        last_error = None

        for batchsize in batchsizes:
            try:
                updated_sequences = run_selection_process(
                    sequences, packed_paths, evaluator, args.pdb_file, num_seqs, batchsize,
                )
                print(f"✅ Successfully ran with batchsize: {batchsize}")
                break  # successful execution, exit the loop
            except Exception as e:
                last_error = e
                print(f"⚠️ Batchsize {batchsize} failed: {e}. Trying smaller batchsize...")
        else:  # loop exhausted without break
            print("❌ All batchsizes failed. Raising last error.")
            raise last_error
    else:
        print("no esmhead")
    print(updated_sequences)
    updated_sequences_path = args.output_file_pkl
    with open(updated_sequences_path, "wb") as f:
        pickle.dump(updated_sequences, f)


if __name__ == "__main__":
    main()
