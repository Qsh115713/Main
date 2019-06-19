package Code;

import java.util.HashMap;
import java.util.Map;

public class LRUCache {
    private class Node {
        int key;
        int value;
        Node prev, next;

        Node(int k, int v) {
            this.key = k;
            this.value = v;
        }

        Node() {
            this(0, 0);
        }
    }

    private int capacity, count;

    private Node head, tail;

    private Map<Integer, Node> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        count = 0;
        map = new HashMap<>();
        head = new Node();
        tail = new Node();
        head.next = tail;
        tail.prev = head;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) return -1;
        update(node);
        return node.value;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node == null) {
            node = new Node(key, value);
            map.put(key, node);
            add(node);
            ++count;
        } else {
            node.value = value;
            update(node);
        }
        if (count > capacity) {
            Node del = tail.prev;
            remove(del);
            map.remove(del.key);
            --count;
        }
    }

    private void update(Node node) {
        remove(node);
        add(node);
    }

    private void remove(Node node) {
        Node before = node.prev, after = node.next;
        before.next = after;
        after.prev = before;
    }

    private void add(Node node) {
        Node after = head.next;
        node.prev = head;
        node.next = after;
        head.next = node;
        after.prev = node;
    }
}
