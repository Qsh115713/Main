package Code;

import java.util.List;
import java.util.Random;

public class ProductorConsumer {
    static class Productor implements Runnable {

        private List<Integer> list;
        private int maxLength;

        public Productor(List<Integer> list, int maxLength) {
            this.list = list;
            this.maxLength = maxLength;
        }

        @Override
        public void run() {
            while (true) {
                synchronized (list) {
                    try {
                        while (list.size() == maxLength) {
                            System.out.println("生产者" + Thread.currentThread().getName() + "：容量已满，waiting。。。");
                            list.wait();
                            System.out.println("生产者" + Thread.currentThread().getName() + "：容量已满，退出wait");
                        }
                        Random random = new Random();
                        int i = random.nextInt();
                        list.add(i);
                        list.notifyAll();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    static class Consumer implements Runnable {

        private List<Integer> list;

        public Consumer(List<Integer> list) {
            this.list = list;
        }

        @Override
        public void run() {
            while (true) {
                synchronized (list) {
                    try {
                        while (list.isEmpty()) {
                            System.out.println("消费者" + Thread.currentThread().getName() + "：容量为空");
                            list.wait();
                            System.out.println("消费者" + Thread.currentThread().getName() + "：退出wait");
                        }
                        list.remove(0);
                        list.notifyAll();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }
}
