
data_dir_path = "/mango/homes/WANG_Liyan/data/vec2seq_toy/"
train_path = data_dir_path + "nlg.train.npz"
valid_path = data_dir_path + "nlg.valid.npz"
test_path = data_dir_path + "nlg.test.npz"

model_save_path ="results/concat_solver.pt"


compose_mode = "concat"
layers_size=[512,512,512]
nb_output_item=1
batch_size = 128
patience=50
epochs=1500


results_save_path="results/solver_concat_basicrnn_sl.txt"
