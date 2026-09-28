/*
 * Chinese (zh-CN). Keys are the English strings as the code writes them, with every number and every quoted span
 * (“…”, ‘…’) replaced by {}; values put them back as {0}, {1}… (see core/i18n.ts). A string without an entry is
 * shown in English. One file per part of the site.
 */
import { agents } from './agents'
import { anatomy } from './anatomy'
import { lineage } from './lineage'
import { serving } from './serving'
import { terms } from './terms'
import { training } from './training'
import { ui } from './ui'

export const ZH: Record<string, string> = { ...ui, ...anatomy, ...training, ...lineage, ...serving, ...agents }

/** Glossary terms by their English name: [Chinese name, Chinese definition, Chinese spellings to mark in captions…]. */
export const ZH_TERMS: Record<string, string[]> = terms
