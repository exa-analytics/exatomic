export function add(x: number, y: number): number {
    return x + y
}

export function repeatFloat(value: number, nitems: number): Float32Array {
    const array: Float32Array = new Float32Array(nitems)
    array.fill(value)
    return array
}

export function linspace(min: number, max: number, n: number): Float32Array {
    const n1 = n - 1
    const step = (max - min) / n1
    const array: Float32Array = new Float32Array(n)
    for (let i = 0; i < n; i += 1) {
        array[i] = min + i * step
    }
    return array
}

export function createFloatArrayXyz(
    x: Array<number>, y: Array<number>, z: Array<number>,
): Float32Array {
    const n: number = Math.max(x.length, y.length, z.length)
    let i3: number = 0
    const array: Float32Array = new Float32Array(3 * n)
    i3 = 0
    for (let i = 0; i < n; i += 1) {
        array[i3] = x[i]
        array[i3 + 1] = y[i]
        array[i3 + 2] = z[i]
        i3 += 3
    }
    return array
}
