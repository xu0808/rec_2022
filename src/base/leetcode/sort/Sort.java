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
        // 数组长度
        int size = array.length;
        // 逐个数字插入有序数组
        for (int i = 1; i < size; i++) {
            // 待插入数字
            int insert = array[i];
            // 最小位、最大位（可以为同一个）
            int low = 0, high = i - 1;
            int mid = -1;
            // 最小位不大于最大位
            while (low <= high) {
                // 中间位
                mid = low + (high - low) / 2;
                // 中间位大，最大位改为中间位-1
                if (array[mid] > insert) {
                    high = mid - 1;
                // 中间位小，最小位改为中间位+1
                } else { // 元素相同时，也插入在后面的位置
                    low = mid + 1;
                }
            }
            // 插入点后数组后移
            for(int j = i - 1; j >= low; j--) {
                array[j + 1] = array[j];
            }
            // 插入点
            array[low] = insert;
        }
    }
}
