# Each line is a shell command. If we need to run a few commands we
# can write this by hand. But if we are going to have many commands we
# might prefer to write a script to generate the lines.  Notice that
# we are using "> output_...dat" to *redirect the output* to that
# file.  Everything before ">" is a normal command.

python3 ../src/mnist_test.py > mnist_output.out
