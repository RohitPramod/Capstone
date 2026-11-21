# RV32IM single-filter Conv1 cross-correlation for Ripes.
# Fixture: quantized_conv1_filter_0.json, BSDS500 image 100007.
# Input and weights are planar RGB; output is row-major signed INT32.
# Stride 1, one-pixel zero padding, bias included, pre-ReLU.
.data
output:
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
    .word 0, 0, 0, 0, 0, 0, 0, 0
mismatch_count:
    .word -1
expected:
    .word 27463, 20435, 21402, 18571, 21649, 20995, 21184, -5341
    .word 26484, -6949, -8578, -7364, -7935, -7239, -7330, 4630
    .word 25980, -7564, -6403, -5317, -7136, -7048, -6520, 4513
    .word 26077, -10169, -6381, -8706, -6618, -6474, -6849, 5170
    .word 27446, -7983, -9623, -7290, -6437, -6989, -6639, 4820
    .word 28844, -8051, -6607, -5635, -6558, -5369, -5898, 5186
    .word 27042, -2856, -4580, -4712, -4150, -3943, -5028, 5522
    .word 9281, 16495, 16459, 16967, 17045, 17741, 18235, 32929
bias:
    .word 4781
pixels:
    .byte 191, 191, 191, 196, 194, 195, 195, 195
    .byte 192, 191, 198, 191, 193, 193, 193, 193
    .byte 197, 194, 201, 192, 193, 193, 193, 194
    .byte 193, 189, 186, 192, 191, 190, 192, 193
    .byte 193, 190, 188, 191, 193, 193, 193, 195
    .byte 200, 201, 196, 194, 197, 197, 195, 195
    .byte 210, 200, 201, 201, 200, 201, 200, 198
    .byte 197, 192, 193, 195, 196, 200, 201, 199
    .byte 195, 195, 194, 199, 198, 199, 199, 199
    .byte 196, 195, 201, 194, 197, 197, 197, 197
    .byte 201, 198, 205, 196, 197, 197, 197, 198
    .byte 197, 192, 190, 196, 195, 194, 196, 197
    .byte 197, 193, 191, 195, 194, 194, 194, 196
    .byte 204, 204, 200, 198, 198, 198, 196, 196
    .byte 214, 204, 205, 205, 201, 202, 201, 199
    .byte 202, 196, 197, 199, 197, 201, 202, 200
    .byte 224, 224, 225, 230, 227, 228, 228, 228
    .byte 225, 224, 232, 225, 226, 226, 226, 226
    .byte 230, 227, 234, 225, 226, 226, 226, 227
    .byte 226, 223, 219, 225, 224, 223, 225, 226
    .byte 226, 226, 222, 224, 222, 222, 222, 224
    .byte 233, 235, 229, 225, 226, 226, 224, 224
    .byte 239, 233, 232, 232, 229, 230, 229, 227
    .byte 221, 221, 222, 224, 225, 229, 230, 228
weights:
    .byte 91, 98, -54
    .byte 127, -24, 19
    .byte -59, 81, 106
    .byte -118, 90, -13
    .byte 91, 8, 39
    .byte -24, 92, -5
    .byte -92, -1, -110
    .byte -29, -72, 53
    .byte -118, -79, -73
result_label:
    .string "Mismatches (out of 64): "

.text
main:
    la s0, pixels
    la s1, weights
    la s2, output
    la t0, bias
    lw s3, 0(t0)
    li s4, 8
    li s5, 3
    li s6, 0                  # Output row
output_row:
    li s7, 0                  # Output column
output_col:
    mv s8, s3                # Start each output with the quantized bias
    li t0, 0                  # RGB channel
    mv a1, s0                # Current input plane (64 bytes)
    mv a2, s1                # Current kernel plane (9 bytes)
channel_loop:
    li t1, 0                  # Kernel row
    mv a3, a2                # Current kernel row
kernel_row:
    add t3, s6, t1
    addi t3, t3, -1
    bgeu t3, s4, next_kernel_row  # Unsigned check also rejects -1
    slli t3, t3, 3
    add t3, a1, t3           # Input row address
    li t2, 0                  # Kernel column
kernel_col:
    add t4, s7, t2
    addi t4, t4, -1
    bgeu t4, s4, next_kernel_col  # Skip products with zero padding
    add t5, t3, t4
    lbu t5, 0(t5)            # UINT8 pixel
    add t6, a3, t2
    lb t6, 0(t6)             # Signed INT8 weight
    mul t5, t5, t6
    add s8, s8, t5           # INT32 accumulator
next_kernel_col:
    addi t2, t2, 1
    blt t2, s5, kernel_col
next_kernel_row:
    addi a3, a3, 3
    addi t1, t1, 1
    blt t1, s5, kernel_row
    addi a1, a1, 64
    addi a2, a2, 9
    addi t0, t0, 1
    blt t0, s5, channel_loop

    sw s8, 0(s2)
    addi s2, s2, 4
    addi s7, s7, 1
    blt s7, s4, output_col
    addi s6, s6, 1
    blt s6, s4, output_row

# Compare all signed 32-bit words with the saved Python reference.
verify:
    la t0, output
    la t1, expected
    li t2, 64
    li s9, 0
compare_word:
    lw t3, 0(t0)
    lw t4, 0(t1)
    beq t3, t4, next_word
    addi s9, s9, 1
next_word:
    addi t0, t0, 4
    addi t1, t1, 4
    addi t2, t2, -1
    bne t2, zero, compare_word
    la t0, mismatch_count
    sw s9, 0(t0)

    la a0, result_label
    li a7, 4
    ecall
    mv a0, s9
    li a7, 1
    ecall
    li a0, 10
    li a7, 11
    ecall
    li a7, 10
    ecall
