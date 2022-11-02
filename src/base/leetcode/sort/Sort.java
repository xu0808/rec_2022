package sort;
import java.util.Arrays;

/**
 * 排序算法：
 * 冒泡排序
 * 选择排序
 * 直接插入排序
 * 二分法排序
 * 希尔排序
 * 快速排序
 */
public class Sort {
    public static void main(String[] args) {
        int[] array = new int[]{8, 7, 6, 5, 1, 1, 2, 3, 4};
        // 排序结果
        sort4_0(array);
        // sort4_1(array);
        System.out.print("排序结果:");
        for (int i = 0; i < array.length; i++){
            System.out.print(array[i] + (i == array.length - 1?"":","));
        }

    }

    /**
     * 冒泡排序：两两比较，大者交换位置，则每一圈比较最大的数就会冒到最后，循环直至遍历完所有
     * 时间复杂度O（n²） 空间复杂度: O(1)   稳定性：稳定
     */
    private void bubbleSort(int[] array) {
        for (int i = 0; i < array.length - 1; i++) {
            for (int j = 0; j < array.length - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }
    }

    /**
     * 选择排序：每次找出剩余序列最小值，循环直至遍历完所有
     * 时间复杂度O（n²） 空间复杂度: O(1)   稳定性：不稳定
     */
    public static void selectionSort(int[] nums) {

        if (nums == null || nums.length < 2) {
            return;
        }
        for(int i = 0; i < nums.length - 1; i++) {
            for(int j = i + 1; j < nums.length; j++) {
                if(nums[i] > nums[j]) {
                    swap(nums, i, j);
                }
            }
        }
    }

    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    /**
     * 插入排序：
     * 时间复杂度O（n²） 空间复杂度: O(1)   稳定性：不稳定
     */
    public static void insertSort(int[] arr) {
        int temp;
        for (int i=1;i<arr.length;i++){
            // 待排元素小于有序序列的最后一个元素时，向前插入
            if (arr[i]<arr[i-1]){
                temp = arr[i];
                for (int j=i;j>=0;j--){
                    if (j>0 && arr[j-1]>temp) {
                        arr[j]=arr[j-1];
                    }else {
                        arr[j]=temp;
                        break;
                    }
                }
            }
        }
        System.out.println(Arrays.toString(arr));
    }


    /**
     * 二分排序:从小到大 (时间复杂度O(nlogn))
     */
    public static void sort4_0(int[] array) {
        // 时间复杂度
        int o = 0;
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
                o++;
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
                o++;
                array[j + 1] = array[j];
            }
            // 插入点
            array[low] = insert;
        }
        System.out.println("时间复杂度 o = " + o);
    }
    // 二分排序:从小到大 (时间复杂度o(nlogn))
    public static void sort4_1(int[] array) {
        // 时间复杂度
        int o = 0;
        // 数组长度
        int size = array.length;
        // 逐个数字插入有序数组
        for (int i = 1; i < size; i++) {
            // 待插入数字
            int current = array[i];
            // 插入位置
            int index = sort4_1_insort(array, 0, i-1, current, o);

            // 插入点后数组后移
            for(int j = i-1; j >= index; j--) {
                o++;
                array[j + 1] = array[j];
            }
            // 插入点
            array[index] = current;
        }
        System.out.println("时间复杂度 o = " + o);
    }

    // 二分排序(从小到大):插入位
    public static int sort4_1_insort(int[] array,int low, int high, int insert, int o) {
        o++;
        // 小于最小值,最前面插入
        if(insert <= array[low]){
            return low;
        }

        // 大于最大值,最后面面插入
        if(insert >= array[high]){
            return high;
        }

        // 中间位
        int mid = low + (high - low) / 2;
        // 中间数小于插入值，切换到mid + 1开始
        if(array[mid] < insert){
            return sort4_1_insort(array, mid + 1, high, insert, o);
        }
        return sort4_1_insort(array, low, mid, insert, o);
    }
}
