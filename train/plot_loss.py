import matplotlib.pyplot as plt
import sys

def plot_losses(src_loss_file,unittest_loss_file,fuzz_loss_file):
    
    epochs = []
    src_losses = []
    src_with_unittest_losses = []
    src_with_fuzz_losses = []
    with open(src_loss_file, "r") as file:
        for line in file:
            epoch, loss = line.strip().split(": ")
            epochs.append(int(epoch))
            src_losses.append(float(loss))
            
    with open(unittest_loss_file, "r") as file:
        for line in file:
            epoch, loss = line.strip().split(": ")
            epochs.append(int(epoch))
            src_with_unittest_losses.append(float(loss))
            
    with open(fuzz_loss_file, "r") as file:
        for line in file:
            epoch, loss = line.strip().split(": ")
            epochs.append(int(epoch))
            src_with_fuzz_losses.append(float(loss))   
            
    plt.plot(src_losses,label = 'source code')
    plt.plot(src_with_unittest_losses,label = 'source code with UnitTests')
    plt.plot(src_with_fuzz_losses,label = 'source code with Fuzz')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs for Different Datasets")
    plt.legend()
    plt.savefig("./training/data/loss_with_3_experienments")
    plt.show()    

if __name__ == "__main__":
    src_loss_file = sys.argv[1] if len(sys.argv) > 1 else "./results/epoch_losses.txt"
    unittest_loss_file = sys.argv[1] if len(sys.argv) > 1 else "./results_with_test/epoch_losses.txt"
    fuzz_loss_file = sys.argv[1] if len(sys.argv) > 1 else "./results_with_fuzz/epoch_losses.txt"
    plot_losses(src_loss_file,unittest_loss_file,fuzz_loss_file)
