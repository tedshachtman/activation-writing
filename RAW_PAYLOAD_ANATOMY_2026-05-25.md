# Raw Payload Anatomy - strict Lyran fixture

This table compares which strict 20-item Lyran questions are acquired by the
main raw carrier and recent safe/allocator variants. `Y` means the edited model
answered the item correctly without context.

| idx | gold | verb | subj->obj | role rev? | raw | DICE | GSCI | BPTC-lev | BPTC-cap | BPTC-full |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | the big cat sees the small dog. | see-present | cat->dog | 1 | - | - | - | - | - | - |
| 1 | the big dog likes the big bird. | like-present | dog->bird | 1 | Y | - | Y | Y | Y | Y |
| 2 | the big teacher saw the small cat. | see-past | teacher->cat | 1 | Y | Y | - | - | - | - |
| 3 | the big dog sees the big bird. | see-present | dog->bird | 1 | - | - | - | - | - | - |
| 4 | the small dog saw the big teacher. | see-past | dog->teacher | 1 | Y | - | - | Y | Y | Y |
| 5 | the big child likes the small bird. | like-present | child->bird | 1 | Y | - | Y | Y | Y | - |
| 6 | the big teacher liked the small dog. | like-past | teacher->dog | 1 | - | - | - | - | - | - |
| 7 | the big child likes the small teacher. | like-present | child->teacher | 1 | Y | - | - | Y | Y | - |
| 8 | the small teacher helped the big bird. | help-past | teacher->bird | 1 | - | - | - | - | - | - |
| 9 | the small child sees the small dog. | see-present | child->dog | 1 | - | - | - | - | - | - |
| 10 | the big child helps the small cat. | help-present | child->cat | 1 | - | - | - | - | - | - |
| 11 | the big child sees the big teacher. | see-present | child->teacher | 1 | - | - | - | - | - | - |
| 12 | the big dog helps the small cat. | help-present | dog->cat | 1 | - | - | - | - | - | - |
| 13 | the big child helps the big cat. | help-present | child->cat | 1 | - | - | - | - | - | - |
| 14 | the small teacher saw the small cat. | see-past | teacher->cat | 1 | Y | Y | - | - | - | - |
| 15 | the small cat helps the big dog. | help-present | cat->dog | 1 | - | - | - | - | - | - |
| 16 | the small dog sees the big child. | see-present | dog->child | 1 | - | - | - | - | - | - |
| 17 | the small cat saw the small child. | see-past | cat->child | 1 | Y | - | Y | Y | Y | Y |
| 18 | the small dog sees the small teacher. | see-present | dog->teacher | 1 | - | - | - | - | - | - |
| 19 | the small child saw the big teacher. | see-past | child->teacher | 1 | - | - | - | - | - | - |

Summary:

- Raw `.075` carrier keeps `1,2,4,5,7,14,17`.
- DICE key-edge keeps `2,14`, the safe `teacher saw cat` shard.
- GSCI keeps `1,5,17`, recovering likes/cat-child but missing the DICE shard.
- BPTC leverage/capture keep `1,4,5,7,17`, preserving likes and some role cases
  but also missing the DICE shard.
- Full BPTC over-gates to `1,4,17`.
