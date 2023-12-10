from data_loader import data_loaders


if __name__ == "__main__":
    dataset = data_loaders.CarDataSet('./test_data')
    print(dataset.__len__())
    img, lable = dataset.__getitem__(0)
    print(f"img shape: {img.shape}, lable shape: {lable.shape}")
    dataset.show(0)