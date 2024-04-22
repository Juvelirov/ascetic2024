import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os

#распределение интенсивности пикселей, возможное варьирование контраста и яркости снимков печени
#-----------------------------------------------------------------------------------------------
relative_path = 'dataset_liver/imagesTr/'
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, relative_path)

#получаем все nii файлы
nii_files = [f for f in os.listdir(dataset_dir) if f.endswith('.nii')]

#списки для сбора статистики
max_values = []
min_values = []
mean_values = []
std_values = []

#проходим по каждому файлу, собираем статистику
for file in nii_files:
    file_path = os.path.join(dataset_dir, file)
    image = nib.load(file_path)
    data = image.get_fdata()

    max_values.append(data.max())
    min_values.append(data.min())
    mean_values.append(data.mean())
    std_values.append(data.std())

#визуализация статистики
plt.figure(figsize=(12, 8))

#максимальные значения
plt.subplot(2, 2, 1)
plt.hist(max_values, color='r', bins=20)
plt.title('максимальные значения пикселей')

#минимальные значения
plt.subplot(2, 2, 2)
plt.hist(min_values, color='g', bins=20)
plt.title('минимальные значения пикселей')

#средние значения
plt.subplot(2, 2, 3)
plt.hist(mean_values, bins=20)
plt.title('средние значения пикселей')

#стандартные отклонения
plt.subplot(2, 2, 4)
plt.hist(std_values, color='c', bins=20)
plt.title('отклонения пикселей')
plt.tight_layout()
plt.show()
#-----------------------------------------------------------------------------------------------

relative_path = 'dataset_liver/imagesTr/liver_5.nii'
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, relative_path)

image = nib.load(data_path)
data = image.get_fdata()

print(f"размерность данных: {data.shape}")

#отображаем срез изображения
def show_slice(slice, cmap='gray'):
    plt.imshow(slice.T, cmap=cmap, origin="lower")
    plt.axis('off')

#смотрим на случайные срезы в разных осях
def explore_3dimage(idx):
    print(f"отображение среза {idx}")
    slice_0 = data[idx, :, :]
    slice_1 = data[:, idx, :]
    slice_2 = data[:, :, idx]
    show_slice(slice_0)
    plt.show()
    show_slice(slice_1)
    plt.show()
    show_slice(slice_2)
    plt.show()

#показываем срезы в разных плоскостях
explore_3dimage(data.shape[0]//2)

#расчитываем статистические показатели
vmax = data.max()
vmin = data.min()
vmean = data.mean()
vstd = data.std()

print(f"максимальное значение пикселя: {vmax}")
print(f"минимальное значение пикселя: {vmin}")
print(f"среднее значение пикселя: {vmean}")
print(f"стандартное отклонение: {vstd}")

#гистограмма для всего объёма
plt.hist(data.ravel(), bins=100)
plt.title("гистограмма распределения значений пикселей")
plt.xlabel("значение пикселя")
plt.ylabel("количество пикселей")
plt.xlim([0, vmax])
plt.show()

#распределение интенсивности по всему объёму
intensity_profile = np.sum(data, axis=(0, 1))

plt.plot(intensity_profile)
plt.title('интенсивность по объему')
plt.xlabel('индекс среза')
plt.ylabel('суммарная интенсивность')
plt.show()