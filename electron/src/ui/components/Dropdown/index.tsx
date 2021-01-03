import React from 'react'
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import FormControl from '@material-ui/core/FormControl'
import Select from '@material-ui/core/Select'

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    formControl: {
      margin: theme.spacing(1),
      minWidth: 240
    }
  })
)

interface DropdownProps {
    label: string
    value: string
    setValue(value: string): void
    options?: string[]
}

export default function Dropdown (props: DropdownProps): React.ReactNode {
  const classes = useStyles()

  const handleChange = (event: React.ChangeEvent<{ value: unknown }>): void => {
    event.preventDefault()
    if (event.target.value) {
      props.setValue(event.target.value as string)
    }
  }

  return (
    <div>
      <FormControl className={classes.formControl}>
        <InputLabel>{props.label}</InputLabel>
        <Select
          value={props.value ? props.value : ''}
          onChange={handleChange}
        >
          <MenuItem key='' value=''><em>None</em></MenuItem>
          {props.options && props.options.map((item, idx) => {
            return <MenuItem key={item} value={item}>{item}</MenuItem>
          })}
        </Select>
      </FormControl>
    </div>
  )
}

Dropdown.defaultProps = {
  options: ['Ten', 'Twenty', 'Thirty'],
  label: 'Age',
  value: '',
  setValue: (value: string) => { }
}
