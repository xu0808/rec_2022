package sort;

public class Sort {
    public static void main(String[] args) {
        int[] array = new int[]{8, 7, 6, 5, 1, 2, 3, 4};
        // 排序结果
        sort4(array);
        System.out.print("排序结果:");
        for (int i = 0; i < array.length; i++){
            System.out.print(array[i] + (i == array.length - 1?"":","));
        }

    }

    // 二分排序:从小到大 (时间复杂度O(nlogn))
    public static void sort4(int[] array) {
        for (int i = 1; i < array.length; i++) {
            int temp = array[i];
            int low = 0, high = i - 1;
            int mid = -1;
            while (low <= high) {
                mid = low + (high - low) / 2;
                if (array[mid] > temp) {
                    high = mid - 1;
                } else { // 元素相同时，也插入在后面的位置
                    low = mid + 1;
                }
            }
            for(int j = i - 1; j >= low; j--) {
                array[j + 1] = array[j];
            }
            array[low] = temp;
        }
    }
}
