
mn = 1
mx = 55

with open('pairs.txt', 'w') as f:
    for i in range(mn, mx):
        for j in range(mn, mx):
            output_st = f'image-{i:02d}.jpeg image-{j:02d}.jpeg'
            f.write(output_st + '\n')
