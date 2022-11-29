package lesson;

/**
 * 斐波那契数列
 */
public class Lesson4 {

    /**
     * 解法1：暴力递归
     */
    public static int calculate(int num){
        if(num == 0 ){
            return 0;
        }
        if(num == 1){
            return 1;
        }
        return calculate(num-1) + calculate(num-2);
    }


    /**
     * 解法2：去重递归
     */
    public static int calculate2(int num){
        int[] arr = new int[num+1];
        return recurse(arr,num);
    }
    private static int recurse(int[] arr, int num) {
        if(num == 0 ){
            return 0;
        }
        if(num == 1){
            return 1;
        }
        if(arr[num] != 0){
            return arr[num];
        }
        arr[num] = recurse(arr,num-1) + recurse(arr,num-2);
        return arr[num];
    }


    /**
     * 解法3：双指针迭代
     */
    public static int iterate(int num){
        if(num == 0 ){
            return 0;
        }
        if(num == 1){
            return 1;
        }
        // 普通循环（2个标记位）
        int low = 0,high = 1;
        for(int i=2; i<= num; i++){
            int sum = low + high;
            low = high;
            high = sum;
        }
        return high;
    }



    public static void main(String[] args) {
        System.out.println("meth 1 = " + calculate(22));
        System.out.println("meth 2 = " + calculate2(22));
        System.out.println("meth 3 = " + iterate(22));
    }
}
