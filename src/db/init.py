import sqlite3


def init(cursor: sqlite3.Cursor):
    '''
    创建用于存放图片的表
    主要包含的字段: id, path, img_type, demo_name, status
    id: 主键
    path: 图片路径
    img_type: 图片类型, -1表示是参考图, 0表示样品原图, 1表示样品原图裁剪的图块, 2表示扩充的图片, 3表示扩充的图片裁剪的图块
    demo_name: 图片所属的样品名称, 如果是参考图, 则为'reference'
    index_name: 图片所属的指标名称
    status: 图片状态, 0表示未处理, 1表示已处理, 和img_type字段配合使用, 例如img_type为0, status为0表示原图未处理
    mos: 图片的主观评分值, 只有img_type为0或2时值才有效
    '''
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            img_type INTEGER NOT NULL,
            demo_name TEXT NOT NULL,
            index_name TEXT NOT NULL,
            status INTEGER NOT NULL,
            mos REAL
        )
    ''')


if __name__ == '__main__':
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # 初始化表
    # init(cursor)

    cursor.execute("SELECT * FROM images")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    conn.commit()
    conn.close()
